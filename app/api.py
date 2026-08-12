from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag import RAGPipeline


app = FastAPI(
    title="Cost-Efficient RAG API",
    description="A local Retrieval-Augmented Generation API using Ollama.",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INITIALIZE RAG PIPELINE
# ============================================================

rag = RAGPipeline()


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Cost-Efficient RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/query")
def query(request: QueryRequest):

    result = rag.query(
        question=request.question,
        top_k=request.top_k
    )

    return result