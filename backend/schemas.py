# backend/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Patient Schema ---
class MedicalHistory(BaseModel):
    date: str
    diagnosis: str
    treatment: str

class Patient(BaseModel):
    patient_id: str = Field(..., example="P001")
    name: str = Field(..., example="John Doe")
    medical_history: List[MedicalHistory] = Field(default_factory=list)
    concern: Optional[str] = Field(None, example="Fever and cough")
    preferred_doctor: Optional[str] = Field(None, example="D001")

# --- Doctor Schema ---
class AvailabilitySlot(BaseModel):
    date: str
    time: str

class Doctor(BaseModel):
    doctor_id: str = Field(..., example="D001")
    name: str = Field(..., example="Dr. Smith")
    specialization: str = Field(..., example="Cardiologist")
    availability: List[AvailabilitySlot] = Field(default_factory=list)

# --- Appointment Schema ---
class Appointment(BaseModel):
    patient_id: str = Field(..., example="P001")
    doctor_id: str = Field(..., example="D001")
    date: str
    time: str
    reason: Optional[str] = Field(None, example="Follow-up consultation")