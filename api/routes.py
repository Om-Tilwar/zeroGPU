#routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List



router = APIRouter()

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

@router.post("/hackrx/run", response_model=HackRxResponse)
async def run_rag_endpoint(payload: HackRxRequest):
    try:
        print(f"Processing documents from URL: {payload.documents}")
        
        # Import ONLY when needed (lazy import)
        from services.rag_service import process_documents_and_questions
        
        results = await process_documents_and_questions(
            pdf_url=str(payload.documents),
            questions=payload.questions
        )
        return {"answers": list(results.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))