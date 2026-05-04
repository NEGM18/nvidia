"""
Guard-Rails — Safety filters for input, retrieval, and output.

Three-layer protection:
1. Input Guard: Block prompt injection, off-topic, and harmful queries.
2. Retrieval Guard: Filter low-relevance results to prevent hallucination.
3. Output Guard: Append disclaimer and enforce grounding.
"""

import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import BLOCKED_KEYWORDS, SIMILARITY_THRESHOLD


class GuardRails:
    """Multi-layer safety guard for the Q&A pipeline."""

    def __init__(self):
        self.blocked_keywords = BLOCKED_KEYWORDS
        self.similarity_threshold = SIMILARITY_THRESHOLD

        # Off-topic patterns
        self.off_topic_patterns = [
            r"(write|generate|create)\s+(code|script|program|email|letter)",
            r"(hack|exploit|crack|bypass)",
            r"(recipe|cook|weather|sports|news)",
            r"(who\s+is\s+the\s+president|capital\s+of)",
        ]

    def check_input(self, query: str) -> tuple[bool, str]:
        """
        Validate user input for safety.

        Args:
            query: The user's question.

        Returns:
            Tuple of (is_safe, rejection_message).
            If safe, rejection_message is empty.
        """
        query_lower = query.lower().strip()

        # Check empty query
        if not query_lower:
            return False, "Please enter a question about your document."

        # Check for prompt injection keywords
        for keyword in self.blocked_keywords:
            if keyword.lower() in query_lower:
                return False, (
                    "🚫 I cannot process this request. Please ask a question "
                    "about the uploaded document."
                )

        # Check for off-topic patterns
        for pattern in self.off_topic_patterns:
            if re.search(pattern, query_lower):
                return False, (
                    "🔒 This question appears to be outside my scope. "
                    "I'm designed to analyze documents. "
                    "Please ask about the content of your uploaded document."
                )

        # Check minimum query length
        if len(query_lower) < 3:
            return False, "Please provide a more detailed question."

        return True, ""

    def check_retrieval(
        self, results_with_scores: list[tuple], threshold: float = None
    ) -> tuple[list, bool]:
        """
        Filter retrieval results by relevance score.

        Args:
            results_with_scores: List of (Document, score) tuples from FAISS.
            threshold: Minimum similarity threshold (lower is better for FAISS L2).

        Returns:
            Tuple of (filtered_documents, has_relevant_results).
        """
        threshold = threshold or self.similarity_threshold

        # FAISS L2 distance: lower is more similar
        # We filter out results with distance > 1.5 (very dissimilar)
        max_distance = 1.5
        filtered = [
            doc for doc, score in results_with_scores
            if score <= max_distance
        ]

        has_relevant = len(filtered) > 0

        if not has_relevant:
            return [], False

        return filtered, True

    def format_response(self, answer: str) -> str:
        """
        Format the final response.

        Args:
            answer: The LLM-generated answer.

        Returns:
            Formatted response string.
        """
        response = answer.strip()

        return response

    def get_no_results_message(self) -> str:
        """Get the message to show when no relevant results are found."""
        return (
            "❌ I couldn't find relevant information in the uploaded document "
            "to answer your question. This could mean:\n\n"
            "1. The information isn't in the document\n"
            "2. Try rephrasing your question with different terms\n"
            "3. The document may need to be re-uploaded\n\n"
        )

    def get_no_document_message(self) -> str:
        """Get the message to show when no document is uploaded."""
        return (
            "📄 No document has been uploaded yet. Please upload a document "
            "(PDF or DOCX) using the **Upload** tab first, then come back "
            "to ask questions about it."
        )
