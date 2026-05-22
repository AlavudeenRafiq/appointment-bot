from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from pymongo import MongoClient
from bson import ObjectId

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["appointment_bot"]
patients_collection = db["patients"]
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]

router = APIRouter()

# ------------------ Schemas ------------------
class MedicalHistory(BaseModel):
    date: str
    diagnosis: str
    treatment: str

class Patient(BaseModel):
    patient_id: str
    name: str
    medical_history: List[MedicalHistory]
    concern: str
    preferred_doctor: str

class Availability(BaseModel):
    date: str
    time: str

class Doctor(BaseModel):
    doctor_id: str
    name: str
    specialization: str
    availability: List[Availability]

class Appointment(BaseModel):
    patient_id: str
    doctor_id: str
    date: str
    time: str

# ------------------ Routes ------------------
@router.post("/patients")
def add_patient(patient: Patient):
    patients_collection.insert_one(patient.dict())
    return {"message": "Patient added successfully"}

@router.get("/patients/{patient_id}")
def fetch_patient(patient_id: str):
    patient = patients_collection.find_one({"patient_id": patient_id}, {"_id": 0})
    return patient if patient else {"error": "Patient not found"}

@router.post("/doctors")
def add_doctor(doctor: Doctor):
    doctors_collection.insert_one(doctor.dict())
    return {"message": "Doctor added successfully"}

@router.get("/doctors/{doctor_id}")
def fetch_doctor(doctor_id: str):
    doctor = doctors_collection.find_one({"doctor_id": doctor_id}, {"_id": 0})
    return doctor if doctor else {"error": "Doctor not found"}

# ------------------ Appointment Booking ------------------
@router.post("/appointments")
def book_appointment(appointment: Appointment):
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
        slot["date"] == appointment.date and slot["time"] == appointment.time
        for slot in doctor["availability"]
    )
    if not available:
        return {"error": "Doctor not available at this time"}

    # Save appointment
    appointments_collection.insert_one(appointment.dict())
    return {"message": "Appointment booked successfully"}

@router.get("/appointments/{patient_id}")
def list_patient_appointments(patient_id: str):
    appointments = list(
        appointments_collection.find({"patient_id": patient_id}, {"_id": 0})
    )
    if not appointments:
        return {"message": "No appointments found for this patient"}
    return {"appointments": appointments}

@router.get("/appointments/doctor/{doctor_id}")
def list_doctor_appointments(doctor_id: str):
    appointments = list(
        appointments_collection.find({"doctor_id": doctor_id}, {"_id": 0})
    )
    if not appointments:
        return {"message": "No appointments found for this doctor"}
    return {"appointments": appointments}

from bson import ObjectId

@router.delete("/appointments/{appointment_id}")
def cancel_appointment(appointment_id: str):
    # Try to delete appointment by its MongoDB ObjectId
    result = appointments_collection.delete_one({"_id": ObjectId(appointment_id)})
    if result.deleted_count == 0:
        return {"error": "Appointment not found"}
    return {"message": "Appointment cancelled successfully"}


@router.put("/appointments/{appointment_id}")
def reschedule_appointment(appointment_id: str, new_date: str, new_time: str):
    # Find appointment
    appointment = appointments_collection.find_one({"_id": ObjectId(appointment_id)})
    if not appointment:
        return {"error": "Appointment not found"}

    # Check doctor availability
    doctor = doctors_collection.find_one({"doctor_id": appointment["doctor_id"]})
    if not doctor:
        return {"error": "Doctor not found"}

    available = any(
        slot["date"] == new_date and slot["time"] == new_time
        for slot in doctor["availability"]
    )
    if not available:
        return {"error": "Doctor not available at this new time"}

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