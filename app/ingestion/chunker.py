"""
Text Chunker — Splits documents into overlapping chunks for embedding.

Uses RecursiveCharacterTextSplitter for intelligent splitting that respects
sentence and paragraph boundaries.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    documents: list[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[Document]:
    """
    Split documents into overlapping chunks.

    Args:
        documents: List of LangChain Documents from the parser.
        chunk_size: Maximum characters per chunk (default from config).
        chunk_overlap: Overlap between consecutive chunks (default from config).

    Returns:
        List of chunked Documents with preserved metadata + chunk index.
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for citation tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    return chunks


def get_chunk_stats(chunks: list[Document]) -> dict:
    """Return statistics about the chunked documents."""
    lengths = [len(c.page_content) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(lengths) / len(lengths) if lengths else 0,
        "min_chunk_size": min(lengths) if lengths else 0,
        "max_chunk_size": max(lengths) if lengths else 0,
        "total_characters": sum(lengths),
    }
