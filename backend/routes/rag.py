from typing import Optional, Any
from fastapi import APIRouter
from pymongo import MongoClient
from backend.pinecone_helper import query_text
from backend.schemas import Patient, Doctor, Appointment
from backend.prompt import build_prompt
from transformers import pipeline

router = APIRouter()

# --- MongoDB connection ---
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["appointment_bot"]
patients_collection = db["patients"]
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]

# --- Hugging Face generator (distilgpt2 only) ---
generator = pipeline("text-generation", model="distilgpt2")

def local_llm(prompt: str) -> str:
    try:
        result = generator(prompt, max_length=256)
        return result[0]["generated_text"]
    except Exception as e:
        return f"⚠️ Local LLM failed: {str(e)}"

@router.post("/rag")
def rag_pipeline(query: str, patient_id: Optional[str] = None):
    # Step 1: Pinecone retrieval (may return list[str] or None)
    raw_context: Any = query_text(query)  # could be list[str], str, or None

    # Normalize to a single string for the prompt
    if isinstance(raw_context, list):
        # join with a separator; trim and keep it concise
        pinecone_context = " ".join([str(x).strip() for x in raw_context if x])
        pinecone_context = pinecone_context[:4000] or "No context available."
    elif isinstance(raw_context, str):
        pinecone_context = raw_context or "No context available."
    else:
        pinecone_context = "No context available."

    # Step 2: MongoDB lookups
    patient_info = None
    if patient_id:
        patient_info = patients_collection.find_one({"patient_id": patient_id}, {"_id": 0})

    doctor = None
    if "Dr." in query:
        doctor_name = query.split("Dr.")[-1].strip().split()[0]
        doctor = doctors_collection.find_one(
            {"name": {"$regex": doctor_name, "$options": "i"}}, {"_id": 0}
        )

    # Step 3: Appointment booking
    appointment_record = None
    if doctor and patient_info and "book" in query.lower():
        slot = doctor.get("availability", [None])[0]
        if slot:
            appointment_record = {
                "patient_id": patient_info.get("patient_id"),
                "doctor_id": doctor.get("doctor_id"),
                "date": slot.get("date"),
                "time": slot.get("time"),
                "reason": patient_info.get("concern", "General checkup")
            }
            appointments_collection.insert_one(appointment_record)

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

    reply = local_llm(prompt)
    source = "distilgpt2"

    return {
        "query": query,
        "context": combined_context,
        "reply": reply,
        "source": source
    }