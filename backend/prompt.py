from typing import Optional


def build_prompt(query: str, pinecone_context: str,
                 patient_info: Optional[dict] = None,
                 doctor_info: Optional[dict] = None,
                 appointment_record: Optional[dict] = None) -> str:

    """
    Build a natural, empathetic prompt for the chatbot.
    """

    patient_context = "No patient record found."
    if patient_info:
        history_count = len(patient_info.get("medical_history") or [])
        patient_context = (
            f"Patient ID: {patient_info.get('patient_id')}; "
            f"name: {patient_info.get('name')}; "
            f"medical history records: {history_count}; "
            f"current concern: {patient_info.get('concern') or 'None'}; "
            f"preferred doctor: {patient_info.get('preferred_doctor') or 'None'}"
        )

    doctor_context = "No doctor selected."
    if doctor_info:
        doctor_context = (
            f"Doctor ID: {doctor_info.get('doctor_id')}; "
            f"name: {doctor_info.get('name')}; "
            f"specialization: {doctor_info.get('specialization')}"
        )

    appointment_context = "No appointment booked."
    if appointment_record:
        appointment_context = (
            f"Doctor ID: {appointment_record.get('doctor_id')}; "
            f"date: {appointment_record.get('date')}; "
            f"time: {appointment_record.get('time')}"
        )

    return f"""
You are a medical appointment assistant.
Respond to the user in a clear, empathetic, and conversational way.

User query: {query}

Background information (for your reasoning only, do not repeat verbatim):
- Pinecone context: {pinecone_context}
- Patient info: {patient_context}
- Doctor info: {doctor_context}
- Appointment booked: {appointment_context}

Guidelines:
- Use the background only to inform your answer.
- Do not dump raw context back to the user.
- If an appointment was booked, confirm it with doctor name, date, and time.
- If no appointment was booked, suggest helpful next steps.
- Always sound supportive and caring, like a human assistant.
"""
