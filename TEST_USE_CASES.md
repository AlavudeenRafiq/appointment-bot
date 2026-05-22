# Appointment Bot Conversation Test Scripts

Use these as literal conversations in the Streamlit app or as `POST /rag` requests. The assistant wording can vary because an LLM may rewrite safe drafts, but the required facts and behavior must match.

## Setup

Start MongoDB, then start the backend and frontend.

```bash
PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 MONGO_URI=mongodb://localhost:27017 PINECONE_NAMESPACE=triage-v1 python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Seed the current RAG dataset when Pinecone is empty or the PDF changes.

```bash
python -m backend.seed_pinecone
```

Current dataset:

```text
backend/data/basic_medical_triage_rag_dataset.pdf
```

Open:

```text
http://127.0.0.1:8501
```

## Conversation 1: Concern Before Identity

Precondition: no patient context exists in the current chat session.

```text
Patient: I have fever and body pain.
Assistant: Please share your patient ID or your name so I can find or create your record.
```

Expected result: no appointment is created and no default patient such as `P001` is assumed.

## Conversation 2: Existing Patient Gives Identity First

Precondition: patient `P001` exists in MongoDB.

```text
Patient: p001
Assistant: What concern or symptoms would you like help with today?
```

Expected result: patient `P001` is loaded into context, but no appointment is created.

## Conversation 3: Fever Gets RAG Protocols Before Booking

Precondition: patient `P001` exists with medical history, for example `Hypertension`.

```text
Patient: p001
Assistant: What concern or symptoms would you like help with today?

Patient: fever
Assistant: Notes the fever, references the patient medical record, gives fever/respiratory RAG triage guidance, and asks the patient to try safe self-care first if no red flags apply.
```

Expected result:

- patient concern is saved as `fever`
- reply mentions the patient history summary such as `Hypertension`
- reply includes fever protocol content such as rest, fluids, fever-free for 24 hours, `103 F / 39.4 C`, red flags, and emergency escalation
- no appointment is created from the fever message alone

## Conversation 4: Follow-Up Booking Uses Stored Concern

Precondition: patient `P001` has stored concern `fever` and preferred doctor `D_PROTOCOL_TEST`; that doctor has an open slot.

```text
Patient: yes, book an appointment
Assistant: Confirms the appointment with doctor name, date, and time.
```

Expected result: appointment is created with reason `fever`, not `yes`.

## Conversation 5: Booking Requires Concern When No Stored Concern Exists

Precondition: patient `P001` exists but has no current concern.

```text
Patient: p001
Assistant: What concern or symptoms would you like help with today?

Patient: Please book Dr. Smith.
Assistant: What concern or symptoms would you like help with today?
```

Expected result: no appointment is created because the assistant still needs a symptom or reason.

## Conversation 6: Booking With Explicit Doctor And Concern

Precondition: patient `P001` exists. Doctor `Dr. Smith` has an open slot.

```text
Patient: My patient ID is P001. Please book Dr. Smith for chest pain.
Assistant: Confirms the appointment with Dr. Smith, date, and time.
```

Expected result: appointment is created with reason `chest pain`.

## Conversation 7: New Patient Gets Protocol Guidance

Precondition: no patient with name `Test Patient` is needed for this run.

```text
Patient: My name is Test Patient and I have cough.
Assistant: Creates or finds the patient, notes cough, says no prior medical conditions are listed, and gives respiratory RAG guidance before offering appointment help.
```

Expected result: a new patient is created with name `Test Patient`, empty medical history, concern `I have cough`, and no appointment.

## Conversation 8: Name-Only New Patient Then Symptom

Precondition: no patient with name `Fresh Patient` is needed for this run.

```text
Patient: My name is Fresh Patient.
Assistant: What concern or symptoms would you like help with today?

Patient: I have headache and dizziness.
Assistant: Notes headache and dizziness, gives headache/neurologic triage guidance, and does not book yet.
```

Expected result: new patient is created, then its concern is updated from the second message.

## Conversation 9: Oral Lesions Use Skin/Mouth Protocol

Precondition: patient `P001` exists.

```text
Patient: p001
Assistant: What concern or symptoms would you like help with today?

Patient: I am having oral lesions.
Assistant: Gives skin/mouth/allergy-style RAG triage guidance, including same-day care and emergency escalation signs.
```

Expected result: no appointment is created from the symptom alone.

## Conversation 10: Stomach Symptoms Use GI Protocol

Precondition: patient `P001` exists.

```text
Patient: My patient ID is P001 and I have vomiting and diarrhea.
Assistant: Gives stomach/GI RAG triage guidance, including fluids, dehydration, blood in stool or vomit, and escalation signs.
```

Expected result: no appointment is created from the symptom alone.

## Conversation 11: Unknown Symptom Asks For More Details

Precondition: patient `P001` exists.

```text
Patient: My patient ID is P001 and I feel strange.
Assistant: Says it does not have a specific first-aid protocol for that symptom and asks for duration, severity, major conditions, medicines, and red flags.
```

Expected result: no appointment is created from the unclear symptom alone.

## Conversation 12: Already Booked First Slot

Precondition: doctor `Dr. Smith` has two availability slots and the first slot is already booked.

```text
Patient: My patient ID is P001. Please book Dr. Smith for cough.
Assistant: Confirms the appointment.
```

Expected result: appointment is created in the next free slot, not the already booked slot.

## Conversation 13: All Doctor Slots Booked

Precondition: doctor `Dr. Smith` exists, but every availability slot already has an appointment.

```text
Patient: My patient ID is P001. Please book Dr. Smith for cough.
Assistant: Says no open slots are available and suggests another doctor or time.
```

Expected result: no duplicate appointment is created.

## Conversation 14: Empty Doctor Availability

Precondition: doctor `Dr. Empty` exists with `availability: []`.

```text
Patient: My patient ID is P001. Please book Dr. Empty for cough.
Assistant: Responds without crashing and says no open slots are available.
```

Expected result: no appointment is created and no `IndexError` occurs.

## Conversation 15: Unknown Doctor

Precondition: patient `P001` exists. No doctor named `Dr. Unknown` exists.

```text
Patient: My patient ID is P001. Please book Dr. Unknown for fever.
Assistant: Says Dr. Unknown cannot be found and asks the user to check the name or choose another doctor.
```

Expected result: no appointment is created.

## Conversation 16: Privacy Check

Precondition: patient `P001` exists with medical history containing a unique treatment string.

```text
Patient: My patient ID is P001 and I have stomach pain.
Assistant: Gives stomach/GI triage guidance.
```

Expected result: the reply does not echo internal prompt text, raw treatment strings, or labels such as `Background information`, `Patient info`, `Doctor info`, or `Appointment`.

## Conversation 17: Frontend Context Continues After Identity

Precondition: patient `P001` exists. Use the Streamlit chat input only.

```text
Patient: p001
Assistant: What concern or symptoms would you like help with today?

Patient: fever
Assistant: Gives fever protocol guidance using the same patient context.
```

Expected result: the frontend does not require a sidebar patient field and does not hardcode `P001`; it continues with the patient context returned by the backend.

## Conversation 18: Frontend Follow-Up Booking

Precondition: patient `P001` exists, has a preferred doctor with an open slot, and sends symptom before booking.

```text
Patient: p001
Assistant: What concern or symptoms would you like help with today?

Patient: fever
Assistant: Gives fever protocol guidance before booking.

Patient: yes, book an appointment
Assistant: Confirms appointment.
```

Expected result: frontend books with reason `fever`, not `yes`.

## Direct API Conversations

```text
POST /rag?query=I%20have%20fever
Assistant: Please share your patient ID or your name so I can find or create your record.
```

```text
POST /rag?query=p001
Assistant: What concern or symptoms would you like help with today?
```

```text
POST /rag?query=fever&patient_id=P001
Assistant: Gives patient-record-aware fever RAG protocol guidance and does not create an appointment.
```

```text
POST /rag?query=yes,%20book%20an%20appointment&patient_id=P001
Assistant: Books using the stored concern if a preferred doctor has availability.
```

```text
POST /rag?query=My%20patient%20ID%20is%20P001.%20Please%20book%20Dr.%20Smith%20for%20chest%20pain
Assistant: Books with reason chest pain.
```

## Cleanup

After each run, delete temporary records created for test patients, doctors, and appointments. Prefer unique test names and IDs such as `Test Patient`, `Fresh Patient`, `P_API_TEST_*`, `D_PROTOCOL_TEST`, and `D_API_TEST_*`.
