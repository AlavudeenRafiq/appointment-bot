import json
import os
import re
from typing import Optional, Any
from fastapi import APIRouter
from pymongo import MongoClient
from backend.pinecone_helper import query_text
from backend.schemas import Patient, Doctor, Appointment
from transformers import pipeline

router = APIRouter()

# --- MongoDB connection ---
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["appointment_bot"]
patients_collection = db["patients"]
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]

llm_generator = None
llm_load_failed = False

LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-3n-E4B-it")
LLM_TASK = os.getenv("LLM_TASK", "image-text-to-text")
VALID_INTENTS = {
    "identity",
    "symptom_new",
    "symptom_update",
    "appointment_yes",
    "appointment_no",
    "appointment_request",
    "thanks_or_closing",
    "unknown"
}
VALID_APPOINTMENT_INTENTS = {"book", "decline", "none"}

def has_booking_intent(query: str) -> bool:
    query_lower = query.lower()
    return any(term in query_lower for term in ("book", "appointment", "schedule"))

def is_booking_followup_concern(concern: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z]+", concern.lower()))
    return normalized in {
        "",
        "yes",
        "yes please",
        "yes book",
        "yes appointment",
        "please do",
        "sure",
        "sure please",
        "book",
        "book appointment",
        "schedule",
        "schedule appointment"
    }

def normalize_intent(query: str) -> str:
    return " ".join(re.findall(r"[a-z]+", query.lower()))

def is_closing_intent(query: str) -> bool:
    normalized = normalize_intent(query)
    return normalized.startswith("thank") or normalized in {
        "thanks",
        "ok",
        "okay",
        "got it",
        "bye",
        "goodbye"
    }

def is_appointment_decline_intent(query: str) -> bool:
    return normalize_intent(query) in {
        "no",
        "no thanks",
        "no thank you",
        "not now",
        "maybe later",
        "later"
    }

def parse_json_object(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None

def clean_optional_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = " ".join(value.strip(" .,:;?!").split())
    return value or None

def parse_optional_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None

def detect_symptom_text(text: str) -> Optional[str]:
    text_lower = text.lower()
    symptom_groups = (
        ("fever", ("fever", "cold", "cough", "sore throat", "body pain")),
        ("oral lesions", ("oral lesion", "mouth lesion", "oral lesions", "lesion")),
        ("rash", ("rash", "hives", "swelling")),
        ("stomach symptoms", ("vomit", "vomiting", "diarrhea", "stomach", "abdomen", "abdominal")),
        ("headache", ("headache", "dizzy", "dizziness", "faint", "weakness", "numbness")),
        ("breathing or chest symptoms", ("chest", "breathing", "breath", "stroke", "confusion")),
    )
    for symptom, terms in symptom_groups:
        if any(term in text_lower for term in terms):
            return symptom
    return None

def rule_based_message_state(query: str, query_without_patient: str,
                             active_concern: Optional[str] = None) -> dict:
    intent_text = query_without_patient or query
    booking_requested = has_booking_intent(query)
    current_concern = (
        extract_current_concern(query_without_patient)
        if booking_requested else query_without_patient.strip()
    )
    duration_days = extract_duration_days(intent_text)
    symptom = detect_symptom_text(current_concern or active_concern or intent_text)
    appointment_intent = "book" if booking_requested else "none"
    intent = "unknown"

    if extract_patient_id(query) or extract_patient_name(query):
        intent = "identity"
    if current_concern:
        intent = "symptom_update" if active_concern and duration_days else "symptom_new"
    if booking_requested:
        intent = "appointment_request"
    if is_booking_followup_concern(current_concern):
        intent = "appointment_yes"
        appointment_intent = "book"
    if is_appointment_decline_intent(intent_text):
        intent = "appointment_no"
        appointment_intent = "decline"
    if is_closing_intent(intent_text):
        intent = "thanks_or_closing"
        appointment_intent = "none"

    if active_concern and intent == "symptom_update" and symptom and symptom not in current_concern.lower():
        current_concern = f"{active_concern}; {intent_text}"

    return {
        "intent": intent,
        "symptom": symptom,
        "concern": current_concern or None,
        "duration_days": duration_days,
        "temperature": None,
        "appointment_intent": appointment_intent,
        "source": "rule-based-extraction"
    }

def validate_message_state(candidate: Optional[dict], fallback: dict, query: str,
                           query_without_patient: str,
                           active_concern: Optional[str] = None) -> dict:
    if not candidate:
        return fallback

    intent_text = query_without_patient or query
    state = fallback.copy()
    intent = clean_optional_text(candidate.get("intent"))
    appointment_intent = clean_optional_text(candidate.get("appointment_intent"))

    if intent in VALID_INTENTS:
        state["intent"] = intent
    if appointment_intent in VALID_APPOINTMENT_INTENTS:
        state["appointment_intent"] = appointment_intent

    symptom = clean_optional_text(candidate.get("symptom"))
    concern = clean_optional_text(candidate.get("concern"))
    temperature = clean_optional_text(candidate.get("temperature"))
    duration_days = parse_optional_int(candidate.get("duration_days"))

    if symptom:
        state["symptom"] = symptom
    if concern:
        state["concern"] = concern
    elif symptom and state["intent"] in {"symptom_new", "symptom_update", "appointment_request"}:
        state["concern"] = symptom
    if duration_days is not None:
        state["duration_days"] = duration_days
    if temperature:
        state["temperature"] = temperature

    if is_closing_intent(intent_text):
        state["intent"] = "thanks_or_closing"
        state["appointment_intent"] = "none"
        state["concern"] = None
    elif is_appointment_decline_intent(intent_text):
        state["intent"] = "appointment_no"
        state["appointment_intent"] = "decline"
        state["concern"] = None
    elif is_booking_followup_concern(intent_text) and active_concern:
        state["intent"] = "appointment_yes"
        state["appointment_intent"] = "book"
        state["concern"] = active_concern
    elif has_booking_intent(query):
        state["appointment_intent"] = "book"
        if state["intent"] not in {"appointment_yes", "appointment_request"}:
            state["intent"] = "appointment_request"

    if active_concern and state["intent"] == "symptom_update":
        state_concern = state.get("concern") or intent_text
        if not detect_symptom_text(state_concern):
            state["concern"] = f"{active_concern}; {intent_text}"

    state["source"] = "llm-extraction"
    return state

def extract_message_state(query: str, query_without_patient: str,
                          active_concern: Optional[str] = None) -> dict:
    fallback = rule_based_message_state(query, query_without_patient, active_concern)
    prompt = (
        "Extract this medical appointment chat message into JSON only. "
        "Allowed intent values: identity, symptom_new, symptom_update, appointment_yes, appointment_no, "
        "appointment_request, thanks_or_closing, unknown. "
        "Allowed appointment_intent values: book, decline, none. "
        "Use null when a value is unknown. Do not add medical advice.\n"
        f"Previous active concern: {active_concern or 'none'}\n"
        f"Message: {query}\n"
        "JSON keys: intent, symptom, concern, duration_days, temperature, appointment_intent"
    )

    reply = run_llm_prompt(prompt, max_new_tokens=140)
    if not reply:
        return fallback

    parsed = parse_json_object(reply)
    return validate_message_state(parsed, fallback, query, query_without_patient, active_concern)

def extract_doctor_name(query: str) -> Optional[str]:
    match = re.search(r"\bdr\.?\s+([a-z]+)", query, re.IGNORECASE)
    return match.group(1) if match else None

def extract_patient_id(query: str) -> Optional[str]:
    match = re.search(r"\bP\d+\b", query, re.IGNORECASE)
    return match.group(0).upper() if match else None

def extract_patient_name(query: str) -> Optional[str]:
    match = re.search(
        r"\b(?:my name is|i am|i'm|this is)\s+([a-z]+(?:\s+[a-z]+)?)",
        query,
        re.IGNORECASE
    )
    if not match:
        return None

    stop_words = {"i", "and", "am", "have", "having", "with", "for", "please", "book", "schedule"}
    name_parts = []
    for word in match.group(1).split():
        if word.lower() in stop_words:
            break
        name_parts.append(word)
    return " ".join(name_parts).title() or None

def remove_patient_details(query: str) -> str:
    query = re.sub(r"\b(?:my\s+)?(?:patient\s+)?id(?:\s+is)?\s*[:#]?\s*P\d+\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\bP\d+\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(
        r"\b(?:my name is|this is)\s+[a-z]+(?:\s+[a-z]+)?(?=\s+(?:and|i\s+am|i'm|i\s+have|have|having|with|for|please)\b|[.?!,;:]|$)",
        " ",
        query,
        flags=re.IGNORECASE
    )
    query = re.sub(
        r"\b(?:i am|i'm)\s+(?!having\b)[a-z]+(?:\s+[a-z]+)?(?=\s+(?:and|i\s+have|have|having|with|for|please)\b|[.?!,;:]|$)",
        " ",
        query,
        flags=re.IGNORECASE
    )
    query = re.sub(r"^\s*(?:hi|hello|hey)\b[, ]*", " ", query, flags=re.IGNORECASE)
    query = " ".join(query.strip(" .,:;?!").split())
    return re.sub(r"^(?:and\s+)+", "", query, flags=re.IGNORECASE)

def extract_current_concern(query: str) -> str:
    concern = re.sub(r"\bdr\.?\s+[a-z]+", " ", query, flags=re.IGNORECASE)
    for term in ("please", "can", "you", "book", "schedule", "appointment",
                 "an", "a", "with", "for"):
        concern = re.sub(rf"\b{term}\b", " ", concern, flags=re.IGNORECASE)
    return " ".join(concern.strip(" .,:;?!").split())

def create_new_patient(query: str, patient_id: Optional[str] = None,
                       patient_name: Optional[str] = None) -> dict:
    """Create a minimal patient record from chat-provided identity details.

    Example:
        create_new_patient("fever", patient_name="Fresh Patient")
    """

    if not patient_id:
        patient_count = patients_collection.count_documents({})
        patient_id = f"P{patient_count + 1:03}"
        while patients_collection.find_one({"patient_id": patient_id}):
            patient_count += 1
            patient_id = f"P{patient_count + 1:03}"

    patient_record = {
        "patient_id": patient_id,
        "name": patient_name or "New Patient",
        "medical_history": [],
        "concern": query,
        "preferred_doctor": None,
        "current_doctors": []
    }
    patients_collection.insert_one(patient_record.copy())
    return patient_record

def extract_doctor_id(value: Any) -> Optional[str]:
    """Return a doctor ID from a string or a small doctor-like dictionary."""

    if isinstance(value, str):
        return clean_optional_text(value)
    if isinstance(value, dict):
        return clean_optional_text(value.get("doctor_id"))
    return None

def get_continuity_doctor_ids(patient_info: Optional[dict]) -> list:
    """Return doctor IDs that should be tried for continuity of care.

    The order is preferred doctor, current doctors, then doctors from prior
    appointments when no preference is stored.

    Example:
        get_continuity_doctor_ids({"preferred_doctor": "D001",
                                   "current_doctors": ["D002"]})
    """

    if not patient_info:
        return []

    doctor_ids = []
    for value in [patient_info.get("preferred_doctor")]:
        doctor_id = extract_doctor_id(value)
        if doctor_id and doctor_id not in doctor_ids:
            doctor_ids.append(doctor_id)

    current_doctors = patient_info.get("current_doctors") or patient_info.get("care_team") or []
    if isinstance(current_doctors, (str, dict)):
        current_doctors = [current_doctors]
    for value in current_doctors:
        doctor_id = extract_doctor_id(value)
        if doctor_id and doctor_id not in doctor_ids:
            doctor_ids.append(doctor_id)

    if not doctor_ids and patient_info.get("patient_id"):
        prior_appointments = appointments_collection.find(
            {"patient_id": patient_info.get("patient_id")},
            {"_id": 0, "doctor_id": 1}
        )
        for appointment in prior_appointments:
            doctor_id = extract_doctor_id(appointment.get("doctor_id"))
            if doctor_id and doctor_id not in doctor_ids:
                doctor_ids.append(doctor_id)

    return doctor_ids

def find_continuity_doctor(patient_info: Optional[dict]) -> Optional[dict]:
    """Find the first existing doctor from the patient's continuity list."""

    for doctor_id in get_continuity_doctor_ids(patient_info):
        doctor = doctors_collection.find_one({"doctor_id": doctor_id}, {"_id": 0})
        if doctor:
            return doctor
    return None

def remember_current_doctor(patient_info: dict, doctor_id: Optional[str]):
    """Save a booked doctor on the patient record for future continuity."""

    doctor_id = extract_doctor_id(doctor_id)
    patient_id = patient_info.get("patient_id") if patient_info else None
    if not patient_id or not doctor_id:
        return

    current_doctors = patient_info.get("current_doctors") or []
    if isinstance(current_doctors, (str, dict)):
        current_doctors = [current_doctors]
    current_doctor_ids = []
    for value in current_doctors:
        current_doctor_id = extract_doctor_id(value)
        if current_doctor_id and current_doctor_id not in current_doctor_ids:
            current_doctor_ids.append(current_doctor_id)
    if doctor_id not in current_doctor_ids:
        current_doctor_ids.append(doctor_id)
        patient_info["current_doctors"] = current_doctor_ids

    patients_collection.update_one(
        {"patient_id": patient_id},
        {"$addToSet": {"current_doctors": doctor_id}}
    )

def format_day_list(days: list) -> str:
    """Return a short patient-facing label for a list of available days."""

    clean_days = [clean_optional_text(day) for day in days]
    clean_days = [day for day in clean_days if day]
    if not clean_days:
        return ""
    if len(clean_days) == 1:
        return clean_days[0]
    if clean_days == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        return "Monday to Friday"
    if clean_days == ["Saturday", "Sunday"]:
        return "Weekend"
    return ", ".join(clean_days)

def build_appointment_schedules(slot: dict) -> list:
    """Convert a doctor availability entry into possible appointment days."""

    if slot.get("date"):
        schedule = {"date": slot.get("date")}
        if slot.get("time"):
            schedule["time"] = slot.get("time")
        return [schedule]

    day_labels = [clean_optional_text(day) for day in slot.get("days") or []]
    day_labels = [day for day in day_labels if day]
    if not day_labels:
        day_label = clean_optional_text(slot.get("label")) or format_day_list([])
        day_labels = [day_label] if day_label else []

    schedules = []
    for day_label in day_labels:
        schedule = {"day": day_label}
        if slot.get("time"):
            schedule["time"] = slot.get("time")
        schedules.append(schedule)
    return schedules

def build_appointment_conflict_filter(doctor_id: str, schedule: dict) -> dict:
    """Build a MongoDB filter that prevents duplicate appointment bookings."""

    conflict_filter = {"doctor_id": doctor_id}
    if schedule.get("date"):
        conflict_filter["date"] = schedule.get("date")
    elif schedule.get("day"):
        conflict_filter["day"] = schedule.get("day")
    if schedule.get("time"):
        conflict_filter["time"] = schedule.get("time")
    return conflict_filter

def format_appointment_when(appointment_record: dict) -> str:
    """Return appointment timing text for either date/time or day-level records."""

    if appointment_record.get("date") and appointment_record.get("time"):
        return f"on {appointment_record.get('date')} at {appointment_record.get('time')}"
    if appointment_record.get("date"):
        return f"on {appointment_record.get('date')}"
    if appointment_record.get("day") and appointment_record.get("time"):
        return f"on {appointment_record.get('day')} at {appointment_record.get('time')}"
    if appointment_record.get("day"):
        return f"on {appointment_record.get('day')}"
    return "for the next available day"

def get_llm_generator():
    global llm_generator, llm_load_failed

    if llm_generator or llm_load_failed:
        return llm_generator

    try:
        pipeline_kwargs = {}
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        llm_device = os.getenv("LLM_DEVICE")
        if hf_token:
            pipeline_kwargs["token"] = hf_token
        if llm_device:
            pipeline_kwargs["device"] = int(llm_device) if llm_device.lstrip("-").isdigit() else llm_device

        llm_generator = pipeline(LLM_TASK, model=LLM_MODEL, **pipeline_kwargs)
    except Exception:
        llm_load_failed = True
    return llm_generator

def build_llm_input(prompt: str):
    if LLM_TASK == "image-text-to-text":
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return prompt

def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(part for part in parts if part)
    return ""

def extract_generated_text(result: Any) -> str:
    if not result:
        return ""
    item = result[0] if isinstance(result, list) else result
    if not isinstance(item, dict):
        return str(item).strip()

    generated = item.get("generated_text") or item.get("text") or ""
    if isinstance(generated, str):
        return generated.strip()
    if isinstance(generated, list):
        for message in reversed(generated):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return extract_text_content(message.get("content")).strip()
        for message in reversed(generated):
            if isinstance(message, dict):
                text = extract_text_content(message.get("content"))
                if text:
                    return text.strip()
    return ""

def run_llm_prompt(prompt: str, max_new_tokens: int) -> Optional[str]:
    generator = get_llm_generator()
    if not generator:
        return None

    generation_args = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if LLM_TASK == "text-generation":
        generation_args["return_full_text"] = False

    try:
        result = generator(build_llm_input(prompt), **generation_args)
    except Exception:
        return None
    return extract_generated_text(result)

def extract_required_terms(draft_reply: str) -> list:
    terms = []
    concern_match = re.search(r"concern:\s*([^.]+)", draft_reply, re.IGNORECASE)
    if concern_match:
        terms.extend(re.findall(r"[a-z0-9]+", concern_match.group(1).lower()))

    record_match = re.search(r"- Record:\s*(.+?)(?:\n-|$)", draft_reply, re.IGNORECASE | re.DOTALL)
    if record_match:
        record_text = record_match.group(1)
        stopwords = {"and", "are", "the", "to", "use", "not", "listed", "details"}
        for term in re.findall(r"[a-z0-9]+", record_text.lower()):
            if term not in stopwords and (len(term) > 2 or term.isdigit()):
                terms.append(term)

    booking_match = re.search(
        r"with\s+(.+?)\s+(on\s+.+?)\s+for\s+([^.]+)",
        draft_reply,
        re.IGNORECASE
    )
    if booking_match:
        for group in booking_match.groups():
            terms.extend(re.findall(r"[a-z0-9]+", group.lower()))

    draft_lower = draft_reply.lower()
    if "i found your record" in draft_lower:
        terms.append("record")
    if "fever" in draft_lower:
        terms.append("fever")
    if "rest" in draft_lower:
        terms.append("rest")
    if "fluids" in draft_lower:
        terms.append("fluids")
    if "103" in draft_lower:
        terms.append("103")
    if "same-day care" in draft_lower:
        terms.extend(["same", "day", "care"])
    if "skin, mouth, rash" in draft_lower:
        terms.extend(["fever", "swelling", "breathing"])
    if "for stomach symptoms" in draft_lower:
        terms.extend(["fluids", "dehydration", "blood"])
    if "for headache or neurologic symptoms" in draft_lower:
        terms.extend(["headache", "confusion", "weakness"])

    return [term for term in terms if len(term) > 2 or term.isdigit()]

def has_repeated_phrase(reply: str) -> bool:
    phrases = re.findall(r"[^.!?]+[.!?]", reply)
    counts = {}
    for phrase in phrases:
        normalized = " ".join(phrase.lower().split())
        counts[normalized] = counts.get(normalized, 0) + 1
        if counts[normalized] > 2:
            return True
    return False

def extract_concern_text(draft_reply: str) -> Optional[str]:
    concern_match = re.search(r"concern:\s*([^.]+)", draft_reply, re.IGNORECASE)
    return concern_match.group(1).strip() if concern_match else None

def repair_llm_reply(reply: str, draft_reply: str) -> str:
    concern = extract_concern_text(draft_reply)
    if not concern:
        return reply

    reply_lower = reply.lower()
    concern_terms = re.findall(r"[a-z0-9]+", concern.lower())
    if any(term not in reply_lower for term in concern_terms):
        reply = f"I've noted your concern: {concern}. {reply}"

    reply_lower = reply.lower()
    if "urgent" in draft_reply.lower() and "urgent" not in reply_lower:
        reply = (
            f"{reply.rstrip()} If symptoms are severe, worsening, or urgent, "
            "please seek immediate medical care."
        )
    return reply

def is_safe_llm_reply(reply: str, draft_reply: str) -> bool:
    if not reply:
        return False

    blocked_terms = (
        "background information",
        "patient info",
        "doctor info",
        "appointment requests",
        "draft response",
        "response:",
        "assistant:",
        "prompt",
        "medical history",
        "i'd like to book",
        "i would like to book",
        "i want to book",
        "i'm a doctor",
        "i am a doctor",
    )
    reply_lower = reply.lower()
    if any(term in reply_lower for term in blocked_terms):
        return False
    if has_repeated_phrase(reply):
        return False
    if "\n- " in draft_reply and "\n- " not in reply:
        return False

    words = reply.split()
    if not 3 <= len(words) <= 90:
        return False

    reply_lower = reply.lower()
    return all(term in reply_lower for term in extract_required_terms(draft_reply))

def naturalize_reply(draft_reply: str) -> tuple:
    required_terms = ", ".join(extract_required_terms(draft_reply)) or "same facts"
    prompt = (
        "Rewrite this medical appointment assistant response in a warm, natural tone. "
        "Use short patient-facing bullet points, not a paragraph. "
        "Keep the same facts. Do not add diagnosis, treatment, medical history, or extra booking details. "
        f"Your reply must include these exact details: {required_terms}. "
        "Return only the patient-facing reply.\n\n"
        f"Draft response: {draft_reply}"
    )

    reply = run_llm_prompt(prompt, max_new_tokens=90)
    if not reply:
        return draft_reply, f"llm-error-fallback:{LLM_MODEL}"

    reply = repair_llm_reply(reply, draft_reply)
    if not is_safe_llm_reply(reply, draft_reply):
        return draft_reply, f"llm-guard-fallback:{LLM_MODEL}"
    return reply, f"llm:{LLM_MODEL}"

def build_medical_record_note(patient_info: dict) -> str:
    history = patient_info.get("medical_history") or []
    visit_notes = []
    has_treatment_context = False
    has_plan_context = False

    for item in history:
        if not isinstance(item, dict):
            continue

        visit_date = clean_optional_text(item.get("date")) or "date not listed"
        diagnosis = clean_optional_text(item.get("diagnosis")) or "diagnosis not listed"
        treatment = clean_optional_text(item.get("treatment"))
        ongoing_plan = clean_optional_text(
            item.get("ongoing_plan")
            or item.get("care_plan")
            or item.get("health_plan")
            or item.get("plan")
        )

        details = [f"{visit_date}: {diagnosis}"]
        if treatment:
            details.append("treatment details are documented")
            has_treatment_context = True
        if ongoing_plan:
            details.append("an ongoing plan is documented")
            has_plan_context = True
        visit_notes.append("; ".join(details))

    if not visit_notes:
        return "No prior visits, diagnoses, treatments, or ongoing plans are listed."

    scheduling_context = ["diagnoses"]
    if has_treatment_context:
        scheduling_context.append("documented treatments")
    if has_plan_context:
        scheduling_context.append("ongoing plans")
    context_text = ", ".join(scheduling_context)

    return (
        f"Past visits: {' | '.join(visit_notes[:3])}. "
        f"Use {context_text} to guide appointment urgency and doctor fit."
    )

def extract_duration_days(current_concern: str) -> Optional[int]:
    match = re.search(r"\b(\d+)\s+days?\b", current_concern.lower())
    return int(match.group(1)) if match else None

def build_protocol_guidance(current_concern: str, pinecone_context: str) -> str:
    concern_lower = current_concern.lower()

    if any(term in concern_lower for term in ("fever", "cold", "cough", "sore throat", "body pain")):
        duration_days = extract_duration_days(current_concern)
        if duration_days and duration_days >= 3:
            return (
                f"- Triage: Fever for {duration_days} days meets same-day care guidance from the PDF.\n"
                "- First aid: Keep resting, drink fluids, and use OTC medicine only as directed.\n"
                "- Next step: Arrange clinician care today. I can help book an appointment if you want."
            )

        return (
            "- First aid: For fever, rest, drink fluids, and use OTC medicine only as directed if no red flags apply.\n"
            "- Same-day care: 103 F / 39.4 C, several days, dehydration, rash/bruising, urinary pain, or serious illness.\n"
            "- Emergency: Breathing trouble, chest pressure, confusion, seizure, stiff neck, severe headache, purple rash, or not urinating.\n"
            "- Next question: What is your temperature and how many days has it been?"
        )

    if any(term in concern_lower for term in ("oral lesion", "mouth lesion", "lesion", "rash", "hives", "swelling")):
        return (
            "For skin, mouth, rash, or allergy-like symptoms, monitor mild local symptoms only if there is no fever, "
            "severe pain, spreading redness, or swelling. Get same-day care for fever, pus, red streaks, eye or genital "
            "involvement, or fast-spreading symptoms. Get emergency help for breathing trouble, throat tightness, "
            "trouble swallowing, tongue/face swelling, fainting, or severe widespread symptoms."
        )

    if any(term in concern_lower for term in ("vomit", "diarrhea", "stomach", "abdomen", "abdominal", "dehydration")):
        return (
            "For stomach symptoms, use fluids, oral rehydration, and bland foods if symptoms are mild. "
            "Get same-day care for worsening belly pain, repeated vomiting, diarrhea over 2 days, dehydration, blood or pus "
            "in stool, or black stool. Get emergency help for severe pain, vomiting blood, confusion, fainting, or severe weakness."
        )

    if any(term in concern_lower for term in ("headache", "dizzy", "dizziness", "faint", "weakness", "numbness")):
        return (
            "For headache or neurologic symptoms, get emergency help for sudden severe headache, confusion, fainting, seizure, "
            "vision changes, weakness, numbness, trouble speaking, or loss of balance. If there are no red flags but it is new, "
            "worsening, recurring, or affecting daily life, arrange clinician review soon."
        )

    if any(term in concern_lower for term in ("chest", "breathing", "breath", "stroke", "confusion")):
        return (
            "For chest pain, breathing trouble, or stroke-like symptoms, get emergency help now for chest pressure, shortness "
            "of breath, sweating, nausea, fainting, jaw or arm pain, blue lips, confusion, face drooping, weakness, numbness, "
            "trouble speaking, vision loss, or sudden loss of balance."
        )

    return (
        "I do not have a specific first-aid protocol for that symptom in the retrieved dataset. Please tell me the duration, "
        "severity, age, pregnancy status if relevant, major conditions, medicines, and any red flags such as breathing trouble, "
        "chest pain, confusion, severe pain, dehydration, bleeding, or rapidly worsening symptoms."
    )

def build_chat_reply(current_concern: str, patient_info: dict, doctor: Optional[dict],
                     appointment_record: Optional[dict], booking_requested: bool,
                     requested_doctor_name: Optional[str], pinecone_context: str) -> str:
    concern = current_concern or "your concern"
    patient_name = patient_info.get("name") if patient_info else None

    if appointment_record and doctor:
        return (
            f"I've booked your appointment with {doctor.get('name', 'the doctor')} "
            f"{format_appointment_when(appointment_record)} "
            f"for {appointment_record.get('reason')}."
        )

    if booking_requested:
        if doctor:
            return (
                f"I found {doctor.get('name', 'the doctor')}, but there is no open "
                "availability right now. Please try another doctor or a different day."
            )
        if requested_doctor_name:
            return (
                f"I couldn't find Dr. {requested_doctor_name.title()} in the doctor records. "
                "Please check the doctor's name or choose another doctor."
            )
        return "Please share the doctor's name so I can check availability and book the appointment."

    intro = "I've noted your concern"
    if patient_name and patient_name != "New Patient":
        intro = f"I found your record, {patient_name}"

    record_note = build_medical_record_note(patient_info)
    protocol_guidance = build_protocol_guidance(concern, pinecone_context)
    return f"{intro}.\n- Record: {record_note}\n{protocol_guidance}"

def build_closing_reply(patient_info: Optional[dict]) -> str:
    concern = patient_info.get("concern") if patient_info else None
    if concern and concern != "Not provided":
        return (
            "You're welcome. Keep monitoring your symptoms, and if they continue, worsen, "
            "or you want help booking an appointment, I can help."
        )
    return "You're welcome. I'm here if you need help with symptoms or an appointment."

def build_appointment_decline_reply() -> str:
    return (
        "No problem. Continue the guidance if no red flags apply. If symptoms continue "
        "or worsen, I can help book an appointment later."
    )

@router.post("/rag")
def rag_pipeline(query: str, patient_id: Optional[str] = None,
                 patient_name: Optional[str] = None):
    patient_id = patient_id or extract_patient_id(query)
    patient_name = patient_name or extract_patient_name(query)
    query_without_patient = remove_patient_details(query)
    patient_info = None
    if patient_id:
        patient_info = patients_collection.find_one({"patient_id": patient_id}, {"_id": 0})

    active_concern = patient_info.get("concern") if patient_info else None
    message_state = extract_message_state(query, query_without_patient, active_concern)
    intent = message_state.get("intent")
    intent_source = message_state.get("source", "rule-based-extraction")

    if not patient_id and not patient_name and intent == "thanks_or_closing":
        return {
            "query": query,
            "context": {
                "pinecone_context": "No context available.",
                "patient_info": None,
                "doctor_info": None,
                "appointment": None
            },
            "reply": build_closing_reply(None),
            "source": intent_source
        }
    if not patient_id and not patient_name:
        return {
            "query": query,
            "context": {
                "pinecone_context": "No context available.",
                "patient_info": None,
                "doctor_info": None,
                "appointment": None
            },
            "reply": "I can help with that. Please share your patient ID or your name so I can find or create your record.",
            "source": intent_source
        }

    # Step 1: LLM extraction plus validated intent routing
    if intent == "thanks_or_closing":
        patient_model = Patient(**patient_info) if patient_info else None
        return {
            "query": query,
            "context": {
                "pinecone_context": "No context available.",
                "patient_info": patient_model.dict() if patient_model else None,
                "doctor_info": None,
                "appointment": None
            },
            "reply": build_closing_reply(patient_info),
            "source": intent_source
        }

    if patient_info and intent == "appointment_no":
        patient_model = Patient(**patient_info)
        return {
            "query": query,
            "context": {
                "pinecone_context": "No context available.",
                "patient_info": patient_model.dict(),
                "doctor_info": None,
                "appointment": None
            },
            "reply": build_appointment_decline_reply(),
            "source": intent_source
        }

    booking_requested = (
        message_state.get("appointment_intent") == "book"
        or intent in {"appointment_yes", "appointment_request"}
    )
    current_concern = message_state.get("concern") or ""
    if (
        patient_info
        and booking_requested
        and (not current_concern or is_booking_followup_concern(current_concern))
        and patient_info.get("concern")
    ):
        current_concern = patient_info.get("concern")

    update_patient_concern = bool(current_concern) and intent in {
        "symptom_new",
        "symptom_update",
        "appointment_request"
    }
    if booking_requested and is_booking_followup_concern(query_without_patient or query):
        update_patient_concern = False

    if not patient_info:
        patient_info = create_new_patient(current_concern or "Not provided", patient_id, patient_name)
    elif update_patient_concern:
        patient_info["concern"] = current_concern
        patients_collection.update_one(
            {"patient_id": patient_info.get("patient_id")},
            {"$set": {"concern": current_concern}}
        )
    if not current_concern:
        patient_model = Patient(**patient_info) if patient_info else None
        return {
            "query": query,
            "context": {
                "pinecone_context": "No context available.",
                "patient_info": patient_model.dict() if patient_model else None,
                "doctor_info": None,
                "appointment": None
            },
            "reply": "Thanks. What concern or symptoms would you like help with today?",
            "source": intent_source
        }

    # Step 2: Pinecone retrieval (may return list[str] or None)
    raw_context: Any = query_text(current_concern or query_without_patient or query)

    # Normalize to a single string for the prompt
    if isinstance(raw_context, list):
        pinecone_context = " ".join([str(x).strip() for x in raw_context if x])
        pinecone_context = pinecone_context[:1200] or "No context available."
    elif isinstance(raw_context, str):
        pinecone_context = raw_context or "No context available."
    else:
        pinecone_context = "No context available."

    doctor = None
    doctor_name = extract_doctor_name(query)
    if doctor_name:
        doctor = doctors_collection.find_one(
            {"name": {"$regex": doctor_name, "$options": "i"}}, {"_id": 0}
        )
    elif booking_requested:
        doctor = find_continuity_doctor(patient_info)
    if doctor:
        doctor["availability"] = [slot for slot in doctor.get("availability") or [] if slot]

    # Step 3: Appointment booking
    appointment_record = None
    if doctor and patient_info and booking_requested and current_concern:
        for slot in doctor.get("availability") or []:
            if not slot:
                continue
            for schedule in build_appointment_schedules(slot):
                existing = appointments_collection.find_one(
                    build_appointment_conflict_filter(doctor.get("doctor_id"), schedule)
                )
                if existing:
                    continue
                appointment_record = {
                    "patient_id": patient_info.get("patient_id"),
                    "doctor_id": doctor.get("doctor_id"),
                    "reason": current_concern,
                    **schedule
                }
                appointments_collection.insert_one(appointment_record)
                remember_current_doctor(patient_info, doctor.get("doctor_id"))
                break
            if appointment_record:
                break

    # Step 4: Convert to Pydantic models
    patient_model = Patient(**patient_info) if patient_info else None
    doctor_model = Doctor(**doctor) if doctor else None
    appointment_model = Appointment(**appointment_record) if appointment_record else None

    combined_context = {
        "pinecone_context": pinecone_context,
        "patient_info": patient_model.dict() if patient_model else None,
        "doctor_info": doctor_model.dict() if doctor_model else None,
        "appointment": appointment_model.dict() if appointment_model else None,
        "message_state": {
            "intent": intent,
            "symptom": message_state.get("symptom"),
            "duration_days": message_state.get("duration_days"),
            "temperature": message_state.get("temperature"),
            "appointment_intent": message_state.get("appointment_intent")
        }
    }

    draft_reply = build_chat_reply(
        current_concern=current_concern,
        patient_info=patient_info,
        doctor=doctor,
        appointment_record=appointment_record,
        booking_requested=booking_requested,
        requested_doctor_name=doctor_name,
        pinecone_context=pinecone_context
    )
    reply, source = naturalize_reply(draft_reply)

    return {
        "query": query,
        "context": combined_context,
        "reply": reply,
        "source": f"{source};{intent_source}"
    }
