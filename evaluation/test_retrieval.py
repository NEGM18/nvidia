"""
Retrieval Quality Tests — Measures retrieval precision and recall.

Tests whether the vector store retrieves the correct chunks
for a set of known question-answer pairs.
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.vectorstore.store import VectorStoreManager
from app.ingestion.parser import parse_file
from app.ingestion.chunker import chunk_documents


# ──────────────────────────────────────────────
# Test Dataset
# ──────────────────────────────────────────────
TEST_QUERIES = [
    {
        "query": "What are the main benefits of AI in software engineering?",
        "expected_keywords": ["productivity", "code generation", "bug detection"],
        "expected_page_or_section": None,
    },
    {
        "query": "What challenges does AI face in software engineering?",
        "expected_keywords": ["security", "privacy", "bias", "hallucinations"],
        "expected_page_or_section": None,
    },
    {
        "query": "How is AI used for quality assurance?",
        "expected_keywords": ["machine learning", "bug", "vulnerabilities"],
        "expected_page_or_section": None,
    },
    {
        "query": "What does NLP do for documentation?",
        "expected_keywords": ["generate", "up-to-date", "conversational"],
        "expected_page_or_section": None,
    },
]


def evaluate_retrieval(
    store: VectorStoreManager,
    test_queries: list[dict] = None,
    k: int = 4,
) -> dict:
    """
    Evaluate retrieval quality.

    Args:
        store: Initialized VectorStoreManager with documents.
        test_queries: List of test query dicts with expected_keywords.
        k: Number of documents to retrieve per query.

    Returns:
        Dictionary with precision metrics.
    """
    test_queries = test_queries or TEST_QUERIES
    results = []

    for test in test_queries:
        query = test["query"]
        expected_keywords = [kw.lower() for kw in test["expected_keywords"]]

        # Retrieve
        docs = store.search(query, k=k)

        # Check if any retrieved chunk contains expected keywords
        retrieved_text = " ".join(doc.page_content.lower() for doc in docs)

        keywords_found = sum(1 for kw in expected_keywords if kw in retrieved_text)
        keyword_recall = keywords_found / len(expected_keywords) if expected_keywords else 0

        results.append({
            "query": query,
            "keywords_found": keywords_found,
            "keywords_total": len(expected_keywords),
            "keyword_recall": keyword_recall,
            "chunks_retrieved": len(docs),
            "passed": keyword_recall >= 0.5,  # At least 50% of keywords found
        })

    # Aggregate metrics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["passed"])
    avg_recall = sum(r["keyword_recall"] for r in results) / total_tests if total_tests else 0

    return {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": total_tests - passed_tests,
        "pass_rate": passed_tests / total_tests if total_tests else 0,
        "avg_keyword_recall": avg_recall,
        "details": results,
    }


def run_retrieval_test(file_path: str) -> dict:
    """
    Run the full retrieval test on a document.

    Args:
        file_path: Path to the test document.

    Returns:
        Evaluation results dictionary.
    """
    # Parse & chunk
    documents = parse_file(file_path)
    chunks = chunk_documents(documents)

    # Create store
    store = VectorStoreManager()
    store.create_from_documents(chunks)

    # Evaluate
    results = evaluate_retrieval(store)

    return results


if __name__ == "__main__":
    # Run on the generic test document
    test_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_document.txt")
    if os.path.exists(test_file):
        results = run_retrieval_test(test_file)
        print(f"\n{'='*60}")
        print(f"RETRIEVAL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Pass Rate: {results['pass_rate']:.1%} ({results['passed']}/{results['total_tests']})")
        print(f"Avg Keyword Recall: {results['avg_keyword_recall']:.1%}")
        print(f"\nDetails:")
        for r in results["details"]:
            status = "✅" if r["passed"] else "❌"
            print(f"  {status} {r['query'][:50]:<50} — Recall: {r['keyword_recall']:.1%}")
    else:
        print(f"Test file not found: {test_file}")
        print("Please create the sample document first.")
