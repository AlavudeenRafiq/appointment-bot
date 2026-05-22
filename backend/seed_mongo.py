from pymongo import MongoClient
import random
from datetime import date, timedelta

# --- Connect to MongoDB ---
client = MongoClient("mongodb://localhost:27017")
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
    ("Diabetes", "Insulin therapy"),
    ("Hypertension", "Lifestyle changes"),
    ("Asthma", "Inhaler"),
    ("Migraine", "Pain management"),
    ("Arthritis", "Physiotherapy"),
    ("Skin Allergy", "Antihistamines"),
    ("Depression", "Counseling"),
    ("Thyroid Disorder", "Medication")
]

# --- Generate doctors ---
doctor_records = []
for i in range(1, 51):
    doctor_id = f"D{i:03}"
    availability = []
    for j in range(3):  # 3 slots per doctor
        slot_date = date(2026, 5, 22) + timedelta(days=j)
        slot_time = f"{9 + (i+j) % 8}:00"
        availability.append({"date": str(slot_date), "time": slot_time})
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
        cond, treat = random.choice(conditions)
        history.append({
            "date": f"2026-0{random.randint(1,5)}-{random.randint(10,28)}",
            "diagnosis": cond,
            "treatment": treat
        })
    patient_records.append({
        "patient_id": patient_id,
        "name": f"Patient{i}",
        "medical_history": history,
        "concern": f"Concern{i}",
        "preferred_doctor": random.choice(doctor_records)["doctor_id"]
    })
patients.insert_many(patient_records)

print("Seeded 50 patients and 50 doctors. Appointments collection is empty.")