"""
API Routes — FastAPI endpoints for the Smart Contract Assistant.

Endpoints:
  POST /api/ingest     — Upload and process a document
  POST /api/chat       — Ask a question about the uploaded document
  POST /api/summarize  — Get a summary of the uploaded document
  GET  /api/status     — Check system status
  POST /api/clear      — Clear the current session
"""

import os
import sys
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import UPLOAD_PATH, DISCLAIMER

from app.ingestion.parser import parse_file
from app.ingestion.chunker import chunk_documents, get_chunk_stats
from app.vectorstore.store import VectorStoreManager
from app.chains.rag_chain import build_rag_chain
from app.chains.conversation import get_session_history, clear_session
from app.chains.summarizer import summarize_documents
from app.guardrails.safety import GuardRails

router = APIRouter(prefix="/api", tags=["Contract Assistant"])

# ──────────────────────────────────────────────
# Shared State (single-user local deployment)
# ──────────────────────────────────────────────
_store_manager = VectorStoreManager()
_guardrails = GuardRails()
_current_documents = []  # Stores the original parsed documents
_current_file_name = ""
_rag_chain = None


# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    disclaimer: str = DISCLAIMER


class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks: int
    stats: dict


class SummarizeResponse(BaseModel):
    summary: str
    disclaimer: str = DISCLAIMER


class StatusResponse(BaseModel):
    has_document: bool
    document_name: str
    chunk_count: int
    ready: bool


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF, DOCX, or TXT)."""
    global _current_documents, _current_file_name, _rag_chain

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save uploaded file
    file_path = UPLOAD_PATH / file.filename
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        # Step 1: Parse
        documents = parse_file(str(file_path))
        if not documents:
            raise HTTPException(status_code=400, detail="No text could be extracted from the file.")

        # Step 2: Chunk
        chunks = chunk_documents(documents)
        stats = get_chunk_stats(chunks)

        # Step 3: Create vector store
        _store_manager.create_from_documents(chunks)
        _store_manager.save()

        # Step 4: Build RAG chain
        retriever = _store_manager.get_retriever()
        _rag_chain = build_rag_chain(retriever)

        # Update state
        _current_documents = chunks
        _current_file_name = file.filename

        return IngestResponse(
            message=f"✅ Successfully processed '{file.filename}'",
            filename=file.filename,
            chunks=len(chunks),
            stats=stats,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about the uploaded document."""
    global _rag_chain

    # Guard-rail: check if document is uploaded
    if not _store_manager.is_ready or _rag_chain is None:
        return ChatResponse(
            answer=_guardrails.get_no_document_message(),
            sources=[],
        )

    # Guard-rail: validate input
    is_safe, rejection_msg = _guardrails.check_input(request.question)
    if not is_safe:
        return ChatResponse(answer=rejection_msg, sources=[])

    try:
        # Get retrieval results with scores for guard-rail check
        results_with_scores = _store_manager.search_with_scores(request.question)
        filtered_docs, has_relevant = _guardrails.check_retrieval(results_with_scores)

        if not has_relevant:
            return ChatResponse(
                answer=_guardrails.get_no_results_message(),
                sources=[],
            )

        # Get conversation history
        history = get_session_history(request.session_id)

        # Run the RAG chain
        answer = _rag_chain.invoke(
            {"question": request.question, "history": history.messages}
        )

        # Add to history
        history.add_user_message(request.question)
        history.add_ai_message(answer)

        # Extract source info
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "chunk_index": doc.metadata.get("chunk_index", "N/A"),
            }
            for doc in filtered_docs
        ]

        formatted_answer = _guardrails.format_response(answer)

        return ChatResponse(answer=formatted_answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize():
    """Generate a summary of the uploaded document."""
    if not _current_documents:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded. Please upload a document first.",
        )

    try:
        summary = summarize_documents(_current_documents)
        return SummarizeResponse(
            summary=_guardrails.format_response(summary),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Check the current system status."""
    return StatusResponse(
        has_document=_store_manager.is_ready,
        document_name=_current_file_name or "None",
        chunk_count=_store_manager.doc_count,
        ready=_store_manager.is_ready,
    )


@router.post("/clear")
async def clear_session_data(session_id: str = "default"):
    """Clear the current session and uploaded documents."""
    global _current_documents, _current_file_name, _rag_chain

    _store_manager.clear()
    clear_session(session_id)
    _current_documents = []
    _current_file_name = ""
    _rag_chain = None

    # Clean upload directory
    for f in UPLOAD_PATH.iterdir():
        if f.name != ".gitkeep":
            f.unlink()

    return {"message": "🗑️ Session cleared successfully."}
