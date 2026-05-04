# 📄 Document Q&A Assistant

An AI-powered web application that allows users to upload documents, then interact with them through intelligent question answering with source citations and guard-rails.

Built with **LangChain**, **FAISS**, **Google Gemini**, **FastAPI**, **LangServe**, and **Gradio**.

---

## 🚀 Features

- **📤 Document Upload** — Support for PDF, DOCX, and TXT files
- **🔍 Intelligent Q&A** — Ask questions and get answers grounded in the document
- **📌 Source Citations** — Every answer includes references to source pages/sections
- **🛡️ Guard-Rails** — Multi-layer safety: input validation, retrieval filtering, output grounding
- **📋 Auto-Summarization** — One-click Map-Reduce document summarization
- **💬 Chat History** — Multi-turn conversational context tracking
- **⚡ Fast Retrieval** — FAISS vector store with HuggingFace embeddings
- **📊 Evaluation Pipeline** — LLM-as-a-Judge metrics for quality assurance

---

## 🏗️ Architecture

```
User ──► Gradio UI (Upload/Chat/Summary)
              │
              ▼
         FastAPI + LangServe (/api/*)
              │
     ┌────────┼────────────┐
     ▼        ▼            ▼
  Ingestion  Retrieval  Summarizer
  Pipeline   + RAG      (Map-Reduce)
     │        │
     ▼        ▼
   FAISS    Guard-Rails ──► Gemini LLM
   Vector     (3-layer)
   Store
```

See the [Project Documentation](docs/project_documentation.md) for a comprehensive guide to the system components.
See [docs/architecture.md](docs/architecture.md) for detailed system design.

---

## 📦 Project Structure

```
nvidia/
├── config.py               # Central configuration
├── requirements.txt         # Python dependencies
├── app/
│   ├── main.py              # FastAPI + LangServe entry point
│   ├── ingestion/
│   │   ├── parser.py        # PDF/DOCX text extraction
│   │   └── chunker.py       # Text splitting & embedding
│   ├── vectorstore/
│   │   └── store.py         # FAISS vector store manager
│   ├── chains/
│   │   ├── rag_chain.py     # RAG Q&A pipeline
│   │   ├── conversation.py  # Chat history management
│   │   └── summarizer.py    # Document summarization
│   ├── guardrails/
│   │   └── safety.py        # Input/output safety filters
│   └── api/
│       └── routes.py        # REST API endpoints
├── ui/
│   └── gradio_app.py        # Gradio frontend
├── evaluation/
│   ├── test_retrieval.py    # Retrieval quality tests
│   ├── test_answers.py      # Answer quality evaluation
│   └── eval_report.py       # Report generator
├── data/
│   ├── sample_document.txt  # Synthetic test document
│   ├── uploads/             # User uploads (gitignored)
│   └── vectorstore/         # FAISS index (gitignored)
└── docs/
    ├── project_documentation.md # Comprehensive project guide
    ├── architecture.md      # System architecture
    └── evaluation_report.md # Evaluation results
```

---

## 🔧 Setup & Installation

### Prerequisites

- Python 3.10+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### 1. Clone the Repository

```bash
git clone <repo-url>
cd nvidia
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Edit the `.env` file with your settings:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 5. Run the Application

**Option A: Gradio UI (Recommended for Demo)**
```bash
python ui/gradio_app.py
```
Open `http://localhost:7860` in your browser.

**Option B: FastAPI Backend**
```bash
python -m app.main
```
API docs at `http://localhost:8000/docs`.

---

## 📖 Usage Guide

### 1. Upload a Document
- Go to the **Upload** tab
- Drag & drop a PDF, DOCX, or TXT file
- Click **Process Document**
- Wait for the success confirmation

### 2. Ask Questions
- Switch to the **Chat** tab
- Type your question (e.g., "What are the payment terms?")
- Get answers with source citations
- Ask follow-up questions — the system tracks context

### 3. Get a Summary
- Go to the **Summary** tab
- Click **Generate Summary**
- Receive a structured overview of the entire document

---

## 📊 Evaluation

Run the evaluation pipeline:

```bash
# Retrieval quality test
python evaluation/test_retrieval.py

# Answer quality test (LLM-as-a-Judge)
python evaluation/test_answers.py

# Generate full evaluation report
python evaluation/eval_report.py
```

Results are saved to `docs/evaluation_report.md`.

---

## 🛡️ Guard-Rails

The system implements three layers of protection:

| Layer | What It Does |
|-------|-------------|
| **Input Guard** | Blocks prompt injection, off-topic queries, harmful content |
| **Retrieval Guard** | Filters low-relevance results (FAISS distance > 1.5) |
| **Output Guard** | Enforces grounding via prompt; appends source citations |

---

## 🔮 Future Enhancements

- [ ] Multi-document search and comparison
- [ ] OCR support for scanned PDFs
- [ ] Domain-specific fine-tuned models
- [ ] User authentication and role-based access
- [ ] Docker containerization
- [ ] Cloud deployment (GCP/AWS)
- [ ] Multi-language support

---

## 📝 License

This project is developed for educational purposes as part of the NVIDIA DLI course on LLM pipelines.

---

## ⚠️ Disclaimer

This is an AI-powered tool for informational purposes only. Always verify important information manually.
