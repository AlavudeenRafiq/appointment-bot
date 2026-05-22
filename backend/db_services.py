from pymongo import MongoClient
import os

# Use environment variable for security
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = MongoClient(MONGO_URI)
db = client["appointment_bot"]

patients_collection = db["patients"]
doctors_collection = db["doctors"]

def insert_patient(patient_data: dict):
    return patients_collection.insert_one(patient_data)

def get_patient(patient_id: str):
    return patients_collection.find_one({"patient_id": patient_id})

def insert_doctor(doctor_data: dict):
    return doctors_collection.insert_one(doctor_data)

def get_doctor(doctor_id: str):
    return doctors_collection.find_one({"doctor_id": doctor_id})