from fastapi import FastAPI
from backend.routes import rag   # import your rag router

app = FastAPI()

# include the rag router
app.include_router(rag.router)

@app.get("/")
def health_check():
    return {"status": "Backend running successfully"}
