# Document Q&A Assistant — Project Documentation

## 1. Project Overview
The **Document Q&A Assistant** is an AI-powered application designed to process, analyze, and extract information from documents. By leveraging Retrieval-Augmented Generation (RAG), the system allows users to ask natural language questions about their uploaded files and provides accurate, grounded answers complete with source citations. 

The application is built with flexibility in mind, supporting local LLM inference via Ollama or cloud-based generation via Google Gemini.

## 2. Core Capabilities
- **Multi-Format Ingestion**: Extracts text from PDF, DOCX, and TXT files.
- **Intelligent Chunking**: Splits large documents into semantically coherent, searchable chunks.
- **High-Performance Retrieval**: Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and a FAISS vector store for sub-millisecond similarity search.
- **Context-Aware Q&A**: Maintains chat history to allow for multi-turn conversational follow-ups.
- **Map-Reduce Summarization**: Can synthesize long documents into structured summaries.
- **Safety First**: Implements input guardrails (prompt injection), retrieval guardrails (distance thresholding), and output guardrails (enforced grounding).

## 3. System Components
The project is modularized into several key components:

### 3.1 Ingestion Pipeline (`app/ingestion/`)
- **`parser.py`**: Handles file reading using `PyMuPDF` for PDFs and `python-docx` for Word documents.
- **`chunker.py`**: Uses LangChain's `RecursiveCharacterTextSplitter` to break text into chunks of 1000 characters with a 200-character overlap.

### 3.2 Vector Store (`app/vectorstore/`)
- **`store.py`**: Manages the FAISS index. It handles embedding the document chunks and provides a retriever interface for the RAG chain.

### 3.3 Chains (`app/chains/`)
- **`rag_chain.py`**: The primary Q&A pipeline. It formats retrieved documents, injects chat history, and queries the LLM with a strict grounding prompt.
- **`summarizer.py`**: Implements a Map-Reduce approach for summarizing long documents without exceeding LLM context windows.
- **`conversation.py`**: Manages in-memory chat session histories.

### 3.4 Safety & Guardrails (`app/guardrails/`)
- **`safety.py`**: Evaluates user input against a blacklist of prompt injection phrases and off-topic patterns. It also filters out low-relevance vector search results.

### 3.5 User Interface (`ui/`)
- **`gradio_app.py`**: A responsive, multi-tab Gradio web interface featuring document upload, interactive chat, and one-click summarization.

## 4. Evaluation & Testing
The system includes an automated LLM-as-a-Judge evaluation pipeline located in the `evaluation/` directory:
- **`test_retrieval.py`**: Tests if the vector store accurately retrieves chunks containing expected keywords.
- **`test_answers.py`**: Uses an LLM to grade RAG responses on correctness, groundedness, relevance, and completeness.
- **`eval_report.py`**: Generates a markdown report summarizing the evaluation results.

## 5. Configuration
System behavior is controlled via `config.py` and environment variables.
- **LLM Provider**: Switch between `ollama` (local) and `gemini` (cloud) via the `LLM_PROVIDER` variable.
- **Chunk Sizing**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` based on document complexity.
- **Retrieval Strictness**: Modify `SIMILARITY_THRESHOLD` to control how strict the semantic search matching should be.

## 6. Further Reading
- For a detailed visual diagram of data flow, refer to [Architecture](architecture.md).
- To see the latest automated test results, check the [Evaluation Report](evaluation_report.md).
