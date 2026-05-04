# System Architecture — Smart Contract Q&A Assistant

## Overview

The Smart Contract Q&A Assistant is a Retrieval-Augmented Generation (RAG) application
that allows users to upload legal documents and interact with them through AI-powered
question answering. The system is designed as a modular, single-user local deployment.

## Architecture Layers

### 1. Frontend Layer (Gradio)

The Gradio UI provides four tabs:
- **Upload Tab** — File upload with drag-and-drop, processing feedback
- **Chat Tab** — Conversational Q&A with source citations
- **Summary Tab** — One-click document summarization
- **Settings Tab** — System status and data management

### 2. API Layer (FastAPI + LangServe)

REST endpoints for programmatic access:
- `POST /api/ingest` — Upload and process documents
- `POST /api/chat` — Send questions and receive answers
- `POST /api/summarize` — Generate document summaries
- `GET /api/status` — Check system status
- `POST /api/clear` — Reset session

LangServe provides a `/chain` playground for testing.

### 3. Pipeline Layer

#### Ingestion Pipeline
```
File Upload → Parser (PyMuPDF/python-docx) → Chunker (RecursiveCharacterTextSplitter)
→ Embedder (all-MiniLM-L6-v2) → FAISS Vector Store
```

#### Retrieval + Answer Pipeline
```
User Query → Embedding → FAISS Similarity Search → Guard-Rail Filter
→ Context Formatting → Prompt Template → Gemini LLM → Source Citations → Response
```

#### Summarization Pipeline
```
All Chunks → Map Phase (individual summaries) → Reduce Phase (combined summary)
→ Structured Output
```

### 4. Storage Layer

- **FAISS** — In-memory vector index with disk persistence
- **File System** — Uploaded files stored locally in `data/uploads/`

### 5. Safety Layer (Guard-Rails)

Three-layer protection:
1. **Input**: Keyword blocklist + off-topic pattern detection
2. **Retrieval**: FAISS distance threshold filtering
3. **Output**: Prompt-based grounding + legal disclaimer

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Google Gemini 2.0 Flash | Answer generation, summarization, evaluation |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) | Semantic vector embeddings |
| Vector Store | FAISS | Fast similarity search |
| Framework | LangChain | Chain composition, prompt management |
| API | FastAPI + LangServe | REST endpoints, auto-docs |
| UI | Gradio | Interactive web interface |
| PDF Parser | PyMuPDF (fitz) | Text extraction from PDFs |
| DOCX Parser | python-docx | Text extraction from Word files |

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interface                             │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│   │  Upload   │   │   Chat   │   │ Summary  │   │ Settings │       │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────────┘       │
│        │              │              │                              │
└────────┼──────────────┼──────────────┼──────────────────────────────┘
         │              │              │
    ┌────▼────┐    ┌────▼────┐    ┌───▼────┐
    │ /ingest │    │  /chat  │    │/summa- │
    │         │    │         │    │ rize   │
    └────┬────┘    └────┬────┘    └───┬────┘
         │              │             │
    ┌────▼────┐    ┌────▼────┐    ┌───▼──────┐
    │ Parser  │    │ Guard-  │    │ Map-     │
    │ Chunker │    │ Rails   │    │ Reduce   │
    │ Embedder│    │ Check   │    │ Chain    │
    └────┬────┘    └────┬────┘    └───┬──────┘
         │              │             │
    ┌────▼──────────────▼─────────────▼────┐
    │           FAISS Vector Store          │
    │         (Embeddings + Metadata)       │
    └──────────────────┬───────────────────┘
                       │
                  ┌────▼────┐
                  │ Gemini  │
                  │  LLM    │
                  └─────────┘
```

## Key Design Decisions

1. **Local-first**: All processing happens locally for data privacy
2. **FAISS over ChromaDB**: Simpler setup, no server required, faster for single-user
3. **HuggingFace embeddings**: No API limits, works offline, consistent results
4. **Map-Reduce summarization**: Handles documents of any length without context window limits
5. **Three-layer guard-rails**: Defense in depth against hallucination and misuse
6. **Session-based history**: In-memory for simplicity; extensible to persistent storage
