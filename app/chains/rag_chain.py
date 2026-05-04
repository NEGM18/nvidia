"""
RAG Chain — Retrieval-Augmented Generation pipeline for contract Q&A.

Combines semantic retrieval from the vector store with LLM-based
answer generation, enforcing grounding and source citations.
"""

import os
import sys
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import LLM_PROVIDER, GEMINI_MODEL, GOOGLE_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL


# ──────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a professional document analysis assistant. Your role is to help users understand the contents of their uploaded documents.

**STRICT RULES:**
1. Answer ONLY based on the provided context from the uploaded document.
2. If the user asks for a general summary or explanation of the document, provide a summary of the provided context.
3. If the answer to a specific question is NOT in the context, clearly state: "I cannot find this information in the uploaded document."
4. NEVER fabricate or hallucinate information not present in the context.
5. Always cite your sources using the format [Source: page X] or [Source: section X].
6. Use clear, professional language.
7. Provide brief plain-language explanations of complex terms if needed.
8. If asked to do something outside document analysis, politely decline.

**CONTEXT FROM DOCUMENT:**
{context}

Answer the user's question based on the above context. Include source citations."""


def _get_llm():
    """Get the configured LLM instance."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.2,
            max_output_tokens=2048,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )


def _format_docs(docs) -> str:
    """Format retrieved documents into a context string with source info."""
    formatted = []
    for i, doc in enumerate(docs):
        source_info = []
        if "page" in doc.metadata:
            source_info.append(f"Page {doc.metadata['page']}")
        if "section" in doc.metadata:
            source_info.append(f"Section {doc.metadata['section']}")
        if "source" in doc.metadata:
            source_info.append(f"File: {doc.metadata['source']}")

        source_str = ", ".join(source_info) if source_info else f"Chunk {i+1}"
        formatted.append(f"[Source: {source_str}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


def build_rag_chain(retriever):
    """
    Build the complete RAG chain.

    Args:
        retriever: A LangChain retriever from the vector store.

    Returns:
        A runnable chain that takes a dict with 'question' and optional 'history'
        keys and returns an answer string with citations.
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{question}"),
    ])

    def _extract_question(x):
        """Extract the question string from input (handles both str and dict)."""
        if isinstance(x, dict):
            return x.get("question", "")
        return str(x)

    # The chain: extract question → retrieve → format → prompt → LLM → parse
    chain = (
        {
            "context": RunnableLambda(_extract_question) | retriever | RunnableLambda(_format_docs),
            "question": RunnableLambda(_extract_question),
            "history": lambda x: x.get("history", []) if isinstance(x, dict) else [],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def build_simple_chain():
    """
    Build a simple LLM chain without retrieval (for general questions).
    Used when no documents are uploaded.
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a document analysis assistant. Currently, no documents have been uploaded. "
            "Please ask the user to upload a document (PDF or DOCX) first before asking questions. "
            "You can explain what you can do: analyze documents, answer questions about specific sections, "
            "summarize documents, and identify key information."
        )),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()
