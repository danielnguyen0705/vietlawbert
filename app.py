from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys

# Add src to path so we can import from law_dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "law_dataset", "src"))

from rag.generator import LegalGenerator

app = FastAPI(title="VietLawBERT API", description="API for querying VietLawBERT Hybrid RAG system")

# Initialize generator once at startup
generator = LegalGenerator()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"message": "VietLawBERT API is running. Use POST /query to ask questions."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_vietlawbert(request: QueryRequest):
    """
    Receive a question and return the answer from VietLawBERT.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer = generator.ask(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)