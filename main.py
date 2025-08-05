#main.py
from fastapi import FastAPI
from api.routes import router as hackrx

app = FastAPI()

app.include_router(hackrx, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok"}
