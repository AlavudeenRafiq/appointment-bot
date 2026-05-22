from fastapi import FastAPI
from backend.routes import appointment, rag

app = FastAPI()

app.include_router(rag.router)
app.include_router(appointment.router)

@app.get("/")
def health_check():
    return {"status": "Backend running successfully"}
