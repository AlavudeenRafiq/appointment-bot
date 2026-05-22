# Appointment Bot Conversation Test Scripts

Use these scripts as literal chat conversations in the Streamlit app or as the `query` value for `POST /rag`. The assistant wording may vary because the local model generates text, but the expected behavior after each script should match.

## Setup

Start MongoDB, then start the backend and frontend.

```bash
PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 MONGO_URI=mongodb://localhost:27017 python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`.

## Conversation 1: User Gives Concern Before Identity

Precondition: No patient context exists in the current chat session.

```text
Patient: I have fever and body pain.
Assistant: Please share your patient ID or your name so I can find or create your record.
```

Expected result: no appointment is created yet, and no default patient such as `P001` is assumed.

## Conversation 2: Existing Patient Shares Current Concern

Precondition: patient `P001` exists in MongoDB.

```text
Patient: My patient ID is P001 and I have fever since yesterday.
Assistant: Responds to the fever concern using the patient context.
```

Expected result: patient `P001` is found, the current concern is saved as `I have fever since yesterday`, and no appointment is created unless the user asks to book one.

## Conversation 3: Existing Patient Gives Identity First

Precondition: patient `P001` exists in MongoDB.

```text
Patient: My patient ID is P001.
Assistant: What concern or symptoms would you like help with today?

Patient: I have a sore throat.
Assistant: Responds to the sore throat concern using the patient context.
```

Expected result: patient `P001` is found after the first message, then the second message updates the current concern to `I have a sore throat`.

## Conversation 4: New Patient Gives Name And Concern

Precondition: no patient with the generated test name is already needed for this run.

```text
Patient: My name is Test Patient and I have cough.
Assistant: Responds to the cough concern after creating or finding the patient record.
```

Expected result: a new patient record is created with name `Test Patient`, empty medical history, and concern `I have cough`.

## Conversation 5: New Patient Gives Name First

Precondition: no patient with the generated test name is already needed for this run.

```text
Patient: My name is Fresh Patient.
Assistant: What concern or symptoms would you like help with today?

Patient: I have headache and nausea.
Assistant: Responds to the headache and nausea concern.
```

Expected result: a new patient record is created, then its concern is updated from the second message.

## Conversation 6: Booking Request Without Concern

Precondition: patient `P001` exists, and doctor `Dr. Smith` exists.

```text
Patient: My patient ID is P001. Please book Dr. Smith.
Assistant: What concern or symptoms would you like help with today?
```

Expected result: no appointment is created because the assistant still needs the reason for the visit.

## Conversation 7: Booking Request With Current Concern

Precondition: patient `P001` exists. Doctor `Dr. Smith` exists with at least one free availability slot.

```text
Patient: My patient ID is P001. Please book Dr. Smith for chest pain.
Assistant: Responds with an appointment booking confirmation or appointment-related response.
```

Expected result: one appointment is created for `P001`, using `Dr. Smith` and reason `chest pain`.

## Conversation 8: Stored Old Concern Must Not Be Reused

Precondition: patient `P001` exists and has an old stored concern such as `old knee pain`. Doctor `Dr. Smith` has a free availability slot.

```text
Patient: My patient ID is P001. Please book Dr. Smith for sore throat.
Assistant: Responds with an appointment booking confirmation or appointment-related response.
```

Expected result: the new appointment reason is `sore throat`, not the old stored concern.

## Conversation 9: Already Booked First Slot

Precondition: doctor `Dr. Smith` has two availability slots, and the first slot already has an appointment.

```text
Patient: My patient ID is P001. Please book Dr. Smith for cough.
Assistant: Responds with an appointment booking confirmation or appointment-related response.
```

Expected result: the appointment is created in the next free slot, not the already booked slot.

## Conversation 10: All Doctor Slots Already Booked

Precondition: doctor `Dr. Smith` exists, but every availability slot already has an appointment.

```text
Patient: My patient ID is P001. Please book Dr. Smith for cough.
Assistant: Responds without crashing.
```

Expected result: no duplicate appointment is created for any already booked slot.

## Conversation 11: Doctor Has Empty Availability

Precondition: doctor `Dr. Empty` exists with `availability: []`.

```text
Patient: My patient ID is P001. Please book Dr. Empty for cough.
Assistant: Responds without crashing.
```

Expected result: no appointment is created, and the backend does not raise an `IndexError`.

## Conversation 12: Unknown Doctor

Precondition: patient `P001` exists. No doctor named `Dr. Unknown` exists.

```text
Patient: My patient ID is P001. Please book Dr. Unknown for fever.
Assistant: Responds without crashing.
```

Expected result: no appointment is created because the requested doctor cannot be found.

## Conversation 13: Privacy Check

Precondition: patient `P001` exists with medical history containing a unique diagnosis or treatment string.

```text
Patient: My patient ID is P001 and I have stomach pain.
Assistant: Responds to the stomach pain concern.
```

Expected result: the assistant reply does not echo internal prompt text, raw medical history, or labels such as `Background information`, `Patient info`, `Doctor info`, or `Appointment`.

## Conversation 14: Frontend Context Continues After Identity

Precondition: patient `P001` exists. Use the Streamlit chat input only.

```text
Patient: My patient ID is P001.
Assistant: What concern or symptoms would you like help with today?

Patient: I have fever.
Assistant: Responds to the fever concern using the same patient context.
```

Expected result: the frontend does not require a sidebar patient field and does not hardcode `P001`; it continues with the patient context returned by the backend.

## Conversation 15: Frontend New Patient Flow

Precondition: use a unique patient name for the run.

```text
Patient: My name is Frontend Test and I have cough.
Assistant: Responds to the cough concern after creating or finding the patient record.

Patient: Please book Dr. Smith for cough.
Assistant: Responds with an appointment booking confirmation or appointment-related response.
```

Expected result: the frontend sends the first identity from chat content, then continues with the returned patient context on the second message.

## Direct API Conversations

These are the same conversations as API calls when testing without the frontend.

```text
POST /rag?query=I%20have%20fever
Assistant: Please share your patient ID or your name so I can find or create your record.
```

```text
POST /rag?query=My%20patient%20ID%20is%20P001%20and%20I%20have%20fever
Assistant: Responds to the fever concern using patient P001.
```

```text
POST /rag?query=My%20name%20is%20API%20Patient%20and%20I%20have%20cough
Assistant: Responds to the cough concern after creating a new patient.
```

```text
POST /rag?query=My%20patient%20ID%20is%20P001.%20Please%20book%20Dr.%20Smith%20for%20chest%20pain
Assistant: Responds with an appointment booking confirmation or appointment-related response.
```

## Cleanup

After each run, delete temporary records created for test patients, doctors, and appointments. Prefer unique test names and IDs such as `API Patient`, `Frontend Test`, `P_API_TEST_*`, and `D_API_TEST_*`.
