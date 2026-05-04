"""
Answer Quality Tests — LLM-as-a-Judge evaluation for RAG answers.

Uses a judge LLM to evaluate:
  - Correctness: Is the answer factually correct?
  - Groundedness: Is the answer based only on the provided context?
  - Relevance: Does the answer address the question?
  - Completeness: Does the answer cover all relevant aspects?
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LLM_PROVIDER, GEMINI_MODEL, GOOGLE_API_KEY

from app.ingestion.parser import parse_file
from app.ingestion.chunker import chunk_documents
from app.vectorstore.store import VectorStoreManager
from app.chains.rag_chain import build_rag_chain


# ──────────────────────────────────────────────
# Evaluation Dataset
# ──────────────────────────────────────────────
EVAL_DATASET = [
    {
        "question": "What is the main purpose of AI in software engineering according to the document?",
        "expected_answer": "AI enhances productivity, automates repetitive tasks, and helps in complex problem-solving.",
    },
    {
        "question": "How do AI tools like GitHub Copilot help developers?",
        "expected_answer": "They assist by suggesting code snippets, completing functions, and writing entire blocks of code based on natural language prompts.",
    },
    {
        "question": "What is a potential risk or challenge mentioned regarding AI models?",
        "expected_answer": "Challenges include security and privacy issues, bias, and hallucinations (generating incorrect or biased code).",
    },
]


JUDGE_PROMPT = """You are an expert evaluator. Evaluate the AI assistant's answer against the expected answer and context.

Question: {question}
Expected Answer: {expected_answer}
AI Answer: {ai_answer}

Rate each dimension from 1 (worst) to 5 (best):

1. **Correctness** (1-5): Is the answer factually correct compared to the expected answer?
2. **Groundedness** (1-5): Does the answer appear to be based only on document content, not hallucinated?
3. **Relevance** (1-5): Does the answer directly address the question asked?
4. **Completeness** (1-5): Does the answer cover the key points from the expected answer?

Respond ONLY with a JSON object in this exact format:
{{"correctness": X, "groundedness": X, "relevance": X, "completeness": X, "reasoning": "brief explanation"}}"""


def _get_judge_llm():
    """Get a judge LLM for evaluation."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen2.5:7b", temperature=0.0)


def evaluate_answers(file_path: str, eval_dataset: list[dict] = None) -> dict:
    """
    Evaluate answer quality using LLM-as-a-Judge.

    Args:
        file_path: Path to the test document.
        eval_dataset: List of question/expected_answer pairs.

    Returns:
        Evaluation results with per-question scores.
    """
    eval_dataset = eval_dataset or EVAL_DATASET

    # Build pipeline
    documents = parse_file(file_path)
    chunks = chunk_documents(documents)
    store = VectorStoreManager()
    store.create_from_documents(chunks)
    retriever = store.get_retriever()
    rag_chain = build_rag_chain(retriever)
    judge = _get_judge_llm()

    results = []

    for item in eval_dataset:
        question = item["question"]
        expected = item["expected_answer"]

        # Generate answer
        try:
            ai_answer = rag_chain.invoke({"question": question, "history": []})
        except Exception as e:
            ai_answer = f"Error: {str(e)}"

        # Judge the answer
        try:
            judge_input = JUDGE_PROMPT.format(
                question=question,
                expected_answer=expected,
                ai_answer=ai_answer,
            )
            judgment = judge.invoke(judge_input)
            judgment_text = judgment.content if hasattr(judgment, 'content') else str(judgment)

            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', judgment_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                scores = {"correctness": 0, "groundedness": 0, "relevance": 0, "completeness": 0, "reasoning": "Failed to parse"}
        except Exception as e:
            scores = {"correctness": 0, "groundedness": 0, "relevance": 0, "completeness": 0, "reasoning": str(e)}

        results.append({
            "question": question,
            "expected_answer": expected,
            "ai_answer": ai_answer,
            "scores": scores,
        })

    # Aggregate
    metrics = ["correctness", "groundedness", "relevance", "completeness"]
    averages = {}
    for metric in metrics:
        vals = [r["scores"].get(metric, 0) for r in results]
        averages[metric] = sum(vals) / len(vals) if vals else 0

    averages["overall"] = sum(averages.values()) / len(averages) if averages else 0

    return {
        "averages": averages,
        "total_questions": len(results),
        "details": results,
    }


if __name__ == "__main__":
    test_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_document.txt")
    if os.path.exists(test_file):
        print("Running answer quality evaluation...")
        print("This may take a few minutes as each answer is generated and judged.\n")

        results = evaluate_answers(test_file)

        print(f"\n{'='*60}")
        print(f"ANSWER QUALITY EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"\nAverage Scores (out of 5):")
        for metric, score in results["averages"].items():
            bar = "█" * int(score) + "░" * (5 - int(score))
            print(f"  {metric:<15} {bar} {score:.2f}")

        print(f"\nPer-Question Details:")
        for r in results["details"]:
            print(f"\n  Q: {r['question']}")
            print(f"  A: {r['ai_answer'][:100]}...")
            s = r["scores"]
            print(f"  Scores: C={s.get('correctness',0)} G={s.get('groundedness',0)} R={s.get('relevance',0)} Co={s.get('completeness',0)}")
    else:
        print(f"Test file not found: {test_file}")
