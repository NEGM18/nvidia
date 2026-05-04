"""
Central configuration for the Document Q&A Assistant.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_PATH = Path(os.getenv("UPLOAD_PATH", BASE_DIR / "data" / "uploads"))
VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", BASE_DIR / "data" / "vectorstore"))

# Ensure directories exist
UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# LLM Provider
# ──────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "gemini"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 4))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))

# ──────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# ──────────────────────────────────────────────
# Guard-Rails
# ──────────────────────────────────────────────
BLOCKED_KEYWORDS = [
    "ignore previous instructions",
    "forget your instructions",
    "act as",
    "pretend you are",
    "jailbreak",
]
