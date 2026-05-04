"""
FastAPI + LangServe Application Entry Point.

Starts the backend server with:
  - REST API endpoints (/api/*)
  - LangServe chain endpoint (/chain)
  - CORS for Gradio frontend
  - Interactive docs at /docs
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import HOST, PORT
from app.api.routes import router

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="📄 Smart Contract Q&A Assistant",
    description=(
        "Upload contracts (PDF/DOCX) and interact with them via AI-powered "
        "question answering with source citations and guard-rails."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)


# ──────────────────────────────────────────────
# Root endpoint
# ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "app": "Smart Contract Q&A Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ingest": "POST /api/ingest",
            "chat": "POST /api/chat",
            "summarize": "POST /api/summarize",
            "status": "GET /api/status",
            "clear": "POST /api/clear",
        },
    }


# ──────────────────────────────────────────────
# LangServe Integration (optional)
# ──────────────────────────────────────────────
try:
    from langserve import add_routes
    from app.chains.rag_chain import build_simple_chain

    # Add a LangServe playground for the simple chain
    add_routes(
        app,
        build_simple_chain(),
        path="/chain",
        playground_type="default",
    )
except ImportError:
    pass  # LangServe is optional


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )
