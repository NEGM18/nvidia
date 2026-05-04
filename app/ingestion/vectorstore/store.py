"""
Vector Store Manager — Manages FAISS index lifecycle.

Supports creating, loading, saving, searching, and adding documents
to a FAISS vector store with HuggingFace embeddings.
"""

import os
import sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import EMBEDDING_MODEL, VECTORSTORE_PATH, TOP_K


class VectorStoreManager:
    """Manages FAISS vector store operations."""

    def __init__(self, embedding_model: str = None, store_path: str = None):
        """
        Initialize the vector store manager.

        Args:
            embedding_model: HuggingFace model name for embeddings.
            store_path: Directory path for saving/loading the FAISS index.
        """
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        self.store_path = Path(store_path or VECTORSTORE_PATH)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store: FAISS | None = None

    def create_from_documents(self, documents: list[Document]) -> None:
        """Create a new FAISS index from a list of documents."""
        if not documents:
            raise ValueError("Cannot create vector store from empty document list.")
        self.store = FAISS.from_documents(documents, self.embedder)

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to an existing vector store."""
        if self.store is None:
            self.create_from_documents(documents)
        else:
            self.store.add_documents(documents)

    def search(self, query: str, k: int = None) -> list[Document]:
        """
        Perform similarity search.

        Args:
            query: The search query string.
            k: Number of top results to return.

        Returns:
            List of most similar documents.
        """
        if self.store is None:
            raise RuntimeError("Vector store not initialized. Ingest documents first.")
        k = k or TOP_K
        return self.store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = None) -> list[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        if self.store is None:
            raise RuntimeError("Vector store not initialized. Ingest documents first.")
        k = k or TOP_K
        return self.store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = None):
        """Get a LangChain retriever interface for the vector store."""
        if self.store is None:
            raise RuntimeError("Vector store not initialized. Ingest documents first.")
        k = k or TOP_K
        return self.store.as_retriever(search_kwargs={"k": k})

    def save(self, index_name: str = "contract_index") -> str:
        """Save the FAISS index to disk."""
        if self.store is None:
            raise RuntimeError("No vector store to save.")
        save_path = str(self.store_path / index_name)
        self.store.save_local(save_path)
        return save_path

    def load(self, index_name: str = "contract_index") -> bool:
        """
        Load a FAISS index from disk.

        Returns:
            True if loaded successfully, False if index doesn't exist.
        """
        load_path = str(self.store_path / index_name)
        if not os.path.exists(load_path):
            return False
        self.store = FAISS.load_local(
            load_path,
            self.embedder,
            allow_dangerous_deserialization=True,
        )
        return True

    def clear(self) -> None:
        """Clear the current vector store from memory."""
        self.store = None

    @property
    def is_ready(self) -> bool:
        """Check if the vector store is initialized and ready for queries."""
        return self.store is not None

    @property
    def doc_count(self) -> int:
        """Get the number of documents in the store."""
        if self.store is None:
            return 0
        return self.store.index.ntotal
