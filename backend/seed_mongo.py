"""Seed local MongoDB with demo patients, doctors, and empty appointments.

Run this after MongoDB is available when you want fresh demo data.

Example:
    MONGO_URI=mongodb://localhost:27017 python -m backend.seed_mongo
"""

from pymongo import MongoClient
import os
import random

# --- Connect to MongoDB ---
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client["appointment_bot"]

patients = db["patients"]
doctors = db["doctors"]
appointments = db["appointments"]

# --- Clear old data ---
patients.delete_many({})
doctors.delete_many({})
appointments.delete_many({})

# --- Sample data ---
specializations = [
    "Cardiologist", "Endocrinologist", "Pulmonologist",
    "Dermatologist", "Neurologist", "Orthopedic",
    "Psychiatrist", "General Physician"
]

conditions = [
    ("Diabetes", "Insulin therapy", "Glucose monitoring and endocrinology follow-up"),
    ("Hypertension", "Lifestyle changes", "Blood pressure monitoring"),
    ("Asthma", "Inhaler", "Pulmonology action plan"),
    ("Migraine", "Pain management", "Neurology follow-up if recurring"),
    ("Arthritis", "Physiotherapy", "Mobility and pain review"),
    ("Skin Allergy", "Antihistamines", "Avoidance plan and dermatology review"),
    ("Depression", "Counseling", "Behavioral health follow-up"),
    ("Thyroid Disorder", "Medication", "Routine thyroid lab monitoring")
]

availability_patterns = [
    {
        "label": "Monday to Friday",
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    },
    {
        "label": "Wednesday",
        "days": ["Wednesday"]
    },
    {
        "label": "Weekend",
        "days": ["Saturday", "Sunday"]
    },
    {
        "label": "Monday and Thursday",
        "days": ["Monday", "Thursday"]
    }
]

# --- Generate doctors ---
doctor_records = []
for i in range(1, 51):
    doctor_id = f"D{i:03}"
    pattern = availability_patterns[(i - 1) % len(availability_patterns)]
    availability = [{
        "label": pattern["label"],
        "days": list(pattern["days"])
    }]
    doctor_records.append({
        "doctor_id": doctor_id,
        "name": f"Dr. Doctor{i}",
        "specialization": random.choice(specializations),
        "availability": availability
    })
doctors.insert_many(doctor_records)

# --- Generate patients ---
patient_records = []
for i in range(1, 51):
    patient_id = f"P{i:03}"
    history = []
    for j in range(random.randint(1, 3)):  # 1–3 history entries
        cond, treat, plan = random.choice(conditions)
        history.append({
            "date": f"2026-0{random.randint(1,5)}-{random.randint(10,28)}",
            "diagnosis": cond,
            "treatment": treat,
            "ongoing_plan": plan
        })
    continuity_doctors = [
        doctor_records[(i - 1) % len(doctor_records)],
        doctor_records[i % len(doctor_records)]
    ]
    preferred_doctor = continuity_doctors[0]["doctor_id"]
    patient_records.append({
        "patient_id": patient_id,
        "name": f"Patient{i}",
        "medical_history": history,
        "concern": f"Concern{i}",
        "preferred_doctor": preferred_doctor,
        "current_doctors": [doctor["doctor_id"] for doctor in continuity_doctors]
    })
patients.insert_many(patient_records)

print("Seeded 50 patients and 50 doctors. Appointments collection is empty.")
