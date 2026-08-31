import os
import time
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.ingestion.pdf_parser import parse_pdf
from app.retrieval.vector_store import vector_store
from app.agents.graph import run_agentic_pipeline
from app.database.stock_db import init_db
from app.database.seed_stock_db import seed_all
from app.evaluation.observability import tracer

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database schema & seed NVDA stock dataset on startup."""
    init_db()
    seed_all()
    yield

app = FastAPI(
    title="OmniBrain — Agentic Multi-Modal RAG Orchestrator",
    description="Agentic RAG System powered by Gemini Flash, LangGraph, Qdrant, SQLite, and FastAPI.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    doc_name: Optional[str] = "NVIDIA Financial Report"

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    is_in_scope: bool
    is_grounded: bool
    route_selected: Optional[str]
    citations: List[dict]
    execution_steps: List[str]
    retrieved_images: List[dict]
    sql_query: Optional[str]
    sql_result: Optional[List[dict]]
    latency_ms: float

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "system": "OmniBrain Orchestrator",
        "gemini_model": settings.GEMINI_MODEL,
        "qdrant_collection": settings.QDRANT_COLLECTION_NAME,
        "stock_db": settings.STOCK_DB_PATH
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF financial report, parses text & figures, and indexes chunks into Qdrant."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Ingestion: Parse PDF text & images
    parse_result = parse_pdf(save_path)

    # Index text chunks into Qdrant vector store
    chunks_added = vector_store.add_chunks(parse_result["text_chunks"])

    return {
        "message": f"Successfully processed and indexed document '{file.filename}'.",
        "total_pages": parse_result["total_pages"],
        "text_chunks_count": len(parse_result["text_chunks"]),
        "indexed_chunks": chunks_added,
        "extracted_images_count": len(parse_result["extracted_images"])
    }

@app.post("/query", response_model=QueryResponse)
def execute_query(req: QueryRequest):
    """Executes the agentic multi-modal RAG workflow for a user prompt."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()
    
    # Run full LangGraph agentic pipeline
    final_state = run_agentic_pipeline(query=req.query, doc_name=req.doc_name)
    
    latency_ms = round((time.time() - start_time) * 1000, 2)
    tracer.log_execution(query=req.query, state=final_state, start_time=start_time)

    return QueryResponse(
        query=req.query,
        final_answer=final_state.get("final_answer", "No answer generated."),
        is_in_scope=final_state.get("is_in_scope", True),
        is_grounded=final_state.get("is_grounded", True),
        route_selected=final_state.get("route"),
        citations=final_state.get("citations", []),
        execution_steps=final_state.get("execution_steps", []),
        retrieved_images=final_state.get("retrieved_images", []),
        sql_query=final_state.get("sql_query"),
        sql_result=final_state.get("sql_result"),
        latency_ms=latency_ms
    )

@app.get("/documents")
def list_documents():
    """Lists uploaded PDF documents and extracted chart images."""
    docs = [f for f in os.listdir(settings.DOCUMENTS_DIR) if f.endswith(".pdf")]
    images = []
    if os.path.exists(settings.EXTRACTED_IMAGES_DIR):
        images = [f for f in os.listdir(settings.EXTRACTED_IMAGES_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]

    return {
        "documents": docs,
        "extracted_images": images
    }
