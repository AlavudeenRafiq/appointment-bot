from fastapi import APIRouter
from pymongo import MongoClient
from bson import ObjectId
import os
from datetime import datetime

from backend.schemas import Appointment, DoctorCreate, PatientCreate

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client["appointment_bot"]
patients_collection = db["patients"]
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]

router = APIRouter()

def day_name_from_date(date_text: str):
    """Return weekday name for a YYYY-MM-DD date string."""

    try:
        return datetime.strptime(date_text, "%Y-%m-%d").strftime("%A")
    except (TypeError, ValueError):
        return None

def normalize_day_text(value: str):
    """Normalize a day or day-group label for comparison."""

    if not value:
        return None
    return " ".join(str(value).strip().lower().split())

def availability_matches_request(slot: dict, date: str = None,
                                 time: str = None, day: str = None) -> bool:
    """Check whether a doctor availability slot matches a requested day/date."""

    if slot.get("date") and slot.get("time"):
        return slot.get("date") == date and slot.get("time") == time

    requested_day = day or day_name_from_date(date)
    requested_day = normalize_day_text(requested_day)
    if not requested_day:
        return False

    slot_days = [normalize_day_text(slot_day) for slot_day in slot.get("days") or []]
    slot_label = normalize_day_text(slot.get("label"))
    return requested_day in slot_days or requested_day == slot_label

def appointment_conflict_filter(appointment: Appointment) -> dict:
    """Build the MongoDB filter used to prevent duplicate bookings."""

    conflict_filter = {"doctor_id": appointment.doctor_id}
    if appointment.date:
        conflict_filter["date"] = appointment.date
    elif appointment.day:
        conflict_filter["day"] = appointment.day
    if appointment.time:
        conflict_filter["time"] = appointment.time
    return conflict_filter

# ------------------ Routes ------------------
@router.post("/patients")
def add_patient(patient: PatientCreate):
    """Create a patient record.

    Example payload:
        {"patient_id": "P001", "name": "John Doe", "medical_history": [],
         "concern": "fever", "preferred_doctor": "D001"}
    """

    patients_collection.insert_one(patient.model_dump())
    return {"message": "Patient added successfully"}

@router.get("/patients/{patient_id}")
def fetch_patient(patient_id: str):
    """Return one patient record by patient ID."""

    patient = patients_collection.find_one({"patient_id": patient_id}, {"_id": 0})
    return patient if patient else {"error": "Patient not found"}

@router.post("/doctors")
def add_doctor(doctor: DoctorCreate):
    """Create a doctor record with day-level availability.

    Example payload:
        {"doctor_id": "D001", "name": "Dr. Smith",
         "specialization": "General Physician",
         "availability": [{"label": "Monday to Friday",
                           "days": ["Monday", "Tuesday", "Wednesday",
                                    "Thursday", "Friday"]}]}
    """

    doctors_collection.insert_one(doctor.model_dump())
    return {"message": "Doctor added successfully"}

@router.get("/doctors/{doctor_id}")
def fetch_doctor(doctor_id: str):
    """Return one doctor record by doctor ID."""

    doctor = doctors_collection.find_one({"doctor_id": doctor_id}, {"_id": 0})
    return doctor if doctor else {"error": "Doctor not found"}

# ------------------ Appointment Booking ------------------
@router.post("/appointments")
def book_appointment(appointment: Appointment):
    """Book an appointment and remember the doctor for continuity of care.

    Example payload:
        {"patient_id": "P001", "doctor_id": "D001",
         "day": "Wednesday", "reason": "fever"}
    """

    # Check patient exists
    patient = patients_collection.find_one({"patient_id": appointment.patient_id})
    if not patient:
        return {"error": "Patient not found"}

    # Check doctor exists
    doctor = doctors_collection.find_one({"doctor_id": appointment.doctor_id})
    if not doctor:
        return {"error": "Doctor not found"}

    # Check doctor availability
    available = any(
        availability_matches_request(
            slot,
            date=appointment.date,
            time=appointment.time,
            day=appointment.day
        )
        for slot in doctor["availability"]
    )
    if not available:
        return {"error": "Doctor not available on this day or time"}

    existing = appointments_collection.find_one(appointment_conflict_filter(appointment))
    if existing:
        return {"error": "Doctor already booked for this day or time"}

    # Save appointment
    appointments_collection.insert_one(appointment.model_dump())
    patients_collection.update_one(
        {"patient_id": appointment.patient_id},
        {"$addToSet": {"current_doctors": appointment.doctor_id}}
    )
    return {"message": "Appointment booked successfully"}

@router.get("/appointments/{patient_id}")
def list_patient_appointments(patient_id: str):
    """Return all appointments for one patient."""

    appointments = list(
        appointments_collection.find({"patient_id": patient_id}, {"_id": 0})
    )
    if not appointments:
        return {"message": "No appointments found for this patient"}
    return {"appointments": appointments}

@router.get("/appointments/doctor/{doctor_id}")
def list_doctor_appointments(doctor_id: str):
    """Return all appointments for one doctor."""

    appointments = list(
        appointments_collection.find({"doctor_id": doctor_id}, {"_id": 0})
    )
    if not appointments:
        return {"message": "No appointments found for this doctor"}
    return {"appointments": appointments}

@router.delete("/appointments/{appointment_id}")
def cancel_appointment(appointment_id: str):
    """Cancel an appointment by MongoDB ObjectId."""

    # Try to delete appointment by its MongoDB ObjectId
    result = appointments_collection.delete_one({"_id": ObjectId(appointment_id)})
    if result.deleted_count == 0:
        return {"error": "Appointment not found"}
    return {"message": "Appointment cancelled successfully"}


@router.put("/appointments/{appointment_id}")
def reschedule_appointment(appointment_id: str, new_date: str, new_time: str):
    """Move an appointment to a new available date and time."""

    # Find appointment
    appointment = appointments_collection.find_one({"_id": ObjectId(appointment_id)})
    if not appointment:
        return {"error": "Appointment not found"}

    # Check doctor availability
    doctor = doctors_collection.find_one({"doctor_id": appointment["doctor_id"]})
    if not doctor:
        return {"error": "Doctor not found"}

    available = any(
        availability_matches_request(slot, date=new_date, time=new_time)
        for slot in doctor["availability"]
    )
    if not available:
        return {"error": "Doctor not available on this new day or time"}

    # Prevent double booking
    existing = appointments_collection.find_one({
        "doctor_id": appointment["doctor_id"],
        "date": new_date,
        "time": new_time
    })
    if existing:
        return {"error": "Doctor already booked at this time"}

    # Update appointment
    appointments_collection.update_one(
        {"_id": ObjectId(appointment_id)},
        {"$set": {"date": new_date, "time": new_time}}
    )
    return {"message": "Appointment rescheduled successfully"}
