"""
Document Summarizer — Map-Reduce summarization for long contracts.

Uses a two-phase approach:
1. Map: Summarize each chunk independently.
2. Reduce: Combine chunk summaries into a final comprehensive summary.
"""

import os
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import LLM_PROVIDER, GEMINI_MODEL, GOOGLE_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL


# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────

MAP_PROMPT = ChatPromptTemplate.from_template(
    """Summarize the following section of a document concisely.
Focus on: main topics, key details, dates, numbers, and important conclusions.

Section:
{text}

Concise Summary:"""
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    """You are a professional document analyst. Combine these section summaries
into a comprehensive, well-structured summary of the entire document.

Organize the summary logically. Here are some suggested sections (if applicable):
1. **Document Overview** — Subject, purpose, and main audience
2. **Key Findings/Points** — The most important takeaways
3. **Important Details** — Critical dates, figures, or specific information
4. **Action Items/Next Steps** — Any required actions or recommendations
5. **Conclusions** — Overall summary or final thoughts

Section Summaries:
{text}

Comprehensive Document Summary:"""
)


def _get_llm():
    """Get the configured LLM instance."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.3,
            max_output_tokens=4096,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
        )


def summarize_documents(documents: list[Document]) -> str:
    """
    Summarize a list of documents using map-reduce approach.

    Args:
        documents: List of Document chunks from the vector store.

    Returns:
        A comprehensive summary string.
    """
    llm = _get_llm()
    map_chain = MAP_PROMPT | llm | StrOutputParser()
    reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()

    if not documents:
        return "No documents to summarize."

    # If the document is short enough, summarize directly
    total_text = "\n\n".join(doc.page_content for doc in documents)
    if len(total_text) < 3000:
        direct_prompt = ChatPromptTemplate.from_template(
            """Provide a comprehensive summary of this document.
Include the main topics, key takeaways, and important conclusions.

Document:
{text}

Summary:"""
        )
        direct_chain = direct_prompt | llm | StrOutputParser()
        return direct_chain.invoke({"text": total_text})

    # Phase 1: Map — summarize each chunk
    chunk_summaries = []
    for doc in documents:
        summary = map_chain.invoke({"text": doc.page_content})
        chunk_summaries.append(summary)

    # Phase 2: Reduce — combine summaries
    combined_summaries = "\n\n---\n\n".join(
        f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )

    final_summary = reduce_chain.invoke({"text": combined_summaries})
    return final_summary
