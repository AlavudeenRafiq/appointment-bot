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

LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-small")
LLM_TASK = os.getenv("LLM_TASK", "text2text-generation")

def has_booking_intent(query: str) -> bool:
    query_lower = query.lower()
    return any(term in query_lower for term in ("book", "appointment", "schedule"))

def is_booking_followup_concern(concern: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z]+", concern.lower()))
    return normalized in {
        "",
        "yes",
        "yes book",
        "yes appointment",
        "book",
        "book appointment",
        "schedule",
        "schedule appointment"
    }

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
        "preferred_doctor": None
    }
    patients_collection.insert_one(patient_record.copy())
    return patient_record

def get_llm_generator():
    global llm_generator, llm_load_failed

    if llm_generator or llm_load_failed:
        return llm_generator

    try:
        llm_generator = pipeline(LLM_TASK, model=LLM_MODEL)
    except Exception:
        llm_load_failed = True
    return llm_generator

def extract_required_terms(draft_reply: str) -> list:
    terms = []
    concern_match = re.search(r"concern:\s*([^.]+)", draft_reply, re.IGNORECASE)
    if concern_match:
        terms.extend(re.findall(r"[a-z0-9]+", concern_match.group(1).lower()))

    booking_match = re.search(
        r"with\s+(.+?)\s+on\s+([0-9-]+)\s+at\s+([0-9:]+)\s+for\s+([^.]+)",
        draft_reply,
        re.IGNORECASE
    )
    if booking_match:
        for group in booking_match.groups():
            terms.extend(re.findall(r"[a-z0-9]+", group.lower()))

    draft_lower = draft_reply.lower()
    if "for fever" in draft_lower or "fever or cold symptoms" in draft_lower:
        terms.extend(["fever", "rest", "fluids", "103"])
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

    words = reply.split()
    if not 3 <= len(words) <= 90:
        return False

    reply_lower = reply.lower()
    return all(term in reply_lower for term in extract_required_terms(draft_reply))

def naturalize_reply(draft_reply: str) -> tuple:
    generator = get_llm_generator()
    if not generator:
        return draft_reply, "rule-based-fallback"

    required_terms = ", ".join(extract_required_terms(draft_reply)) or "same facts"
    prompt = (
        "Rewrite this medical appointment assistant response in a warm, natural tone. "
        "Use one concise paragraph. "
        "Keep the same facts. Do not add diagnosis, treatment, medical history, or extra booking details. "
        f"Your reply must include these exact details: {required_terms}. "
        "Return only the patient-facing reply.\n\n"
        f"Draft response: {draft_reply}"
    )

    try:
        generation_args = {"max_new_tokens": 90, "do_sample": False}
        if LLM_TASK == "text-generation":
            generation_args["return_full_text"] = False

        result = generator(prompt, **generation_args)
        reply = result[0].get("generated_text", "").strip()
    except Exception:
        return draft_reply, f"llm-error-fallback:{LLM_MODEL}"

    reply = repair_llm_reply(reply, draft_reply)
    if not is_safe_llm_reply(reply, draft_reply):
        return draft_reply, f"llm-guard-fallback:{LLM_MODEL}"
    return reply, f"llm:{LLM_MODEL}"

def build_medical_record_note(patient_info: dict) -> str:
    history = patient_info.get("medical_history") or []
    diagnoses = []
    for item in history:
        diagnosis = item.get("diagnosis") if isinstance(item, dict) else None
        if diagnosis and diagnosis not in diagnoses:
            diagnoses.append(diagnosis)

    if not diagnoses:
        return "I do not see prior medical conditions listed in your record."

    diagnosis_text = ", ".join(diagnoses[:3])
    return f"Your record lists {diagnosis_text}, so please be a little cautious."

def build_protocol_guidance(current_concern: str, pinecone_context: str) -> str:
    concern_lower = current_concern.lower()

    if any(term in concern_lower for term in ("fever", "cold", "cough", "sore throat", "body pain")):
        return (
            "For fever, rest, drink fluids, and use OTC medicine only as directed. "
            "Get same-day care if it reaches 103 F / 39.4 C, lasts several days, or comes with dehydration, rash/bruising, "
            "urinary pain, or serious illness. Get emergency help for breathing trouble, chest pressure, confusion, seizure, "
            "stiff neck, severe headache, purple rash, or not urinating."
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
    display_concern = concern
    concern_lower = concern.lower()
    for known_concern in ("oral lesions", "mouth lesions", "fever", "cough", "headache", "diarrhea", "vomiting"):
        if known_concern in concern_lower:
            display_concern = known_concern
            break
    patient_name = patient_info.get("name") if patient_info else None

    if appointment_record and doctor:
        return (
            f"I've booked your appointment with {doctor.get('name', 'the doctor')} "
            f"on {appointment_record.get('date')} at {appointment_record.get('time')} "
            f"for {appointment_record.get('reason')}."
        )

    if booking_requested:
        if doctor:
            return (
                f"I found {doctor.get('name', 'the doctor')}, but there are no open slots "
                "available right now. Please try another doctor or a different time."
            )
        if requested_doctor_name:
            return (
                f"I couldn't find Dr. {requested_doctor_name.title()} in the doctor records. "
                "Please check the doctor's name or choose another doctor."
            )
        return "Please share the doctor's name so I can check availability and book the appointment."

    intro = "I've noted your concern"
    if patient_name and patient_name != "New Patient":
        intro = f"I found your record, {patient_name}. Concern"

    record_note = build_medical_record_note(patient_info)
    protocol_guidance = build_protocol_guidance(concern, pinecone_context)
    return (
        f"{intro}: {display_concern}. {record_note} {protocol_guidance} "
        "Try these steps if no red flags apply. If it continues or worsens, I can help book an appointment."
    )

@router.post("/rag")
def rag_pipeline(query: str, patient_id: Optional[str] = None,
                 patient_name: Optional[str] = None):
    patient_id = patient_id or extract_patient_id(query)
    patient_name = patient_name or extract_patient_name(query)
    query_without_patient = remove_patient_details(query)
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
            "source": "rule-based"
        }

    # Step 1: Pinecone retrieval (may return list[str] or None)
    raw_context: Any = query_text(query_without_patient or query)  # could be list[str], str, or None

    # Normalize to a single string for the prompt
    if isinstance(raw_context, list):
        # join with a separator; trim and keep it concise
        pinecone_context = " ".join([str(x).strip() for x in raw_context if x])
        pinecone_context = pinecone_context[:1200] or "No context available."
    elif isinstance(raw_context, str):
        pinecone_context = raw_context or "No context available."
    else:
        pinecone_context = "No context available."

    booking_requested = has_booking_intent(query)
    current_concern = (
        extract_current_concern(query_without_patient)
        if booking_requested else query_without_patient.strip()
    )

    # Step 2: MongoDB lookups
    patient_info = None
    if patient_id:
        patient_info = patients_collection.find_one({"patient_id": patient_id}, {"_id": 0})

    update_patient_concern = bool(current_concern)
    if (
        patient_info
        and booking_requested
        and is_booking_followup_concern(current_concern)
        and patient_info.get("concern")
    ):
        current_concern = patient_info.get("concern")
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
                "pinecone_context": pinecone_context,
                "patient_info": patient_model.dict() if patient_model else None,
                "doctor_info": None,
                "appointment": None
            },
            "reply": "Thanks. What concern or symptoms would you like help with today?",
            "source": "rule-based"
        }

    doctor = None
    doctor_name = extract_doctor_name(query)
    if doctor_name:
        doctor = doctors_collection.find_one(
            {"name": {"$regex": doctor_name, "$options": "i"}}, {"_id": 0}
        )
    elif booking_requested and patient_info.get("preferred_doctor"):
        doctor = doctors_collection.find_one(
            {"doctor_id": patient_info.get("preferred_doctor")}, {"_id": 0}
        )
    if doctor:
        doctor["availability"] = [slot for slot in doctor.get("availability") or [] if slot]

    # Step 3: Appointment booking
    appointment_record = None
    if doctor and patient_info and booking_requested and current_concern:
        for slot in doctor.get("availability") or []:
            if not slot:
                continue
            existing = appointments_collection.find_one({
                "doctor_id": doctor.get("doctor_id"),
                "date": slot.get("date"),
                "time": slot.get("time")
            })
            if existing:
                continue
            appointment_record = {
                "patient_id": patient_info.get("patient_id"),
                "doctor_id": doctor.get("doctor_id"),
                "date": slot.get("date"),
                "time": slot.get("time"),
                "reason": current_concern
            }
            appointments_collection.insert_one(appointment_record)
            break

    # Step 4: Convert to Pydantic models
    patient_model = Patient(**patient_info) if patient_info else None
    doctor_model = Doctor(**doctor) if doctor else None
    appointment_model = Appointment(**appointment_record) if appointment_record else None

    combined_context = {
        "pinecone_context": pinecone_context,
        "patient_info": patient_model.dict() if patient_model else None,
        "doctor_info": doctor_model.dict() if doctor_model else None,
        "appointment": appointment_model.dict() if appointment_model else None
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
        "source": source
    }
