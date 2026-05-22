import os
import re
from typing import Optional, Any
from fastapi import APIRouter
from pymongo import MongoClient
from backend.pinecone_helper import query_text
from backend.schemas import Patient, Doctor, Appointment
from backend.prompt import build_prompt
from transformers import pipeline

router = APIRouter()

# --- MongoDB connection ---
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["appointment_bot"]
patients_collection = db["patients"]
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]

# --- Hugging Face generator (distilgpt2 only) ---
generator = pipeline("text-generation", model="distilgpt2")

def has_booking_intent(query: str) -> bool:
    query_lower = query.lower()
    return any(term in query_lower for term in ("book", "appointment", "schedule"))

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

def local_llm(prompt: str) -> str:
    try:
        result = generator(prompt, max_new_tokens=128, return_full_text=False)
        return result[0]["generated_text"].strip()
    except Exception as e:
        return f"⚠️ Local LLM failed: {str(e)}"

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
            "source": "distilgpt2"
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
    if not patient_info:
        patient_info = create_new_patient(current_concern or "Not provided", patient_id, patient_name)
    elif current_concern:
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
            "source": "distilgpt2"
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

    # Step 5: Build prompt using helper (pinecone_context is a str now)
    prompt = build_prompt(
        query=query,
        pinecone_context=pinecone_context,
        patient_info=patient_info,
        doctor_info=doctor,
        appointment_record=appointment_record
    )

    if booking_requested and doctor and not current_concern:
        reply = "Before I book that appointment, what concern or symptoms should I include for the doctor?"
    else:
        reply = local_llm(prompt)
    source = "distilgpt2"

    return {
        "query": query,
        "context": combined_context,
        "reply": reply,
        "source": source
    }
