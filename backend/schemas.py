"""Shared Pydantic models for the appointment bot.

These models define the shape of patient, doctor, and appointment data used by
the FastAPI routes and the RAG flow.

Example:
    appointment = Appointment(
        patient_id="P001",
        doctor_id="D001",
        day="Wednesday",
        reason="fever",
    )
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

# --- Patient Schema ---
class MedicalHistory(BaseModel):
    """One past visit or condition from a patient's medical record."""

    date: str
    diagnosis: str
    treatment: str
    ongoing_plan: Optional[str] = Field(None, example="Monthly follow-up")

class Patient(BaseModel):
    """Patient profile used by chat responses and appointment scheduling."""

    patient_id: str = Field(..., example="P001")
    name: str = Field(..., example="John Doe")
    medical_history: List[MedicalHistory] = Field(default_factory=list)
    concern: Optional[str] = Field(None, example="Fever and cough")
    preferred_doctor: Optional[str] = Field(None, example="D001")
    current_doctors: List[str] = Field(default_factory=list, example=["D001"])

class PatientCreate(Patient):
    """Required payload for creating a patient through the API."""

    medical_history: List[MedicalHistory] = Field(..., example=[])
    concern: str = Field(..., example="Fever and cough")
    preferred_doctor: str = Field(..., example="D001")

# --- Doctor Schema ---
class AvailabilitySlot(BaseModel):
    """Day-level or date/time availability offered by a doctor.

    Example:
        AvailabilitySlot(label="Monday to Friday",
                         days=["Monday", "Tuesday", "Wednesday",
                               "Thursday", "Friday"])
    """

    label: Optional[str] = Field(None, example="Monday to Friday")
    days: List[str] = Field(default_factory=list, example=["Monday", "Tuesday"])
    date: Optional[str] = Field(None, example="2026-05-24")
    time: Optional[str] = Field(None, example="09:00")

    @model_validator(mode="after")
    def day_or_datetime_required(self):
        if self.days or (self.date and self.time):
            return self
        raise ValueError("Availability must include days or date/time")

class Doctor(BaseModel):
    """Doctor profile with specialization and available appointment slots."""

    doctor_id: str = Field(..., example="D001")
    name: str = Field(..., example="Dr. Smith")
    specialization: str = Field(..., example="Cardiologist")
    availability: List[AvailabilitySlot] = Field(default_factory=list)

class DoctorCreate(Doctor):
    """Required payload for creating a doctor through the API."""

    availability: List[AvailabilitySlot] = Field(..., example=[])

# --- Appointment Schema ---
class Appointment(BaseModel):
    """Booked appointment, including the patient's reason for the visit.

    Example:
        Appointment(patient_id="P001", doctor_id="D001",
                    day="Wednesday", reason="fever")
    """

    patient_id: str = Field(..., example="P001")
    doctor_id: str = Field(..., example="D001")
    date: Optional[str] = Field(None, example="2026-05-24")
    time: Optional[str] = Field(None, example="09:00")
    day: Optional[str] = Field(None, example="Wednesday")
    reason: str = Field(..., min_length=1, example="Follow-up consultation")

    @model_validator(mode="after")
    def day_or_date_required(self):
        if self.day or self.date:
            return self
        raise ValueError("Appointment day or date is required")

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, reason: str) -> str:
        reason = reason.strip()
        if not reason:
            raise ValueError("Appointment reason is required")
        return reason
