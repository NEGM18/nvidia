"""
Evaluation Report Generator — Combines retrieval and answer metrics
into a formatted markdown report.
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.test_retrieval import run_retrieval_test
from evaluation.test_answers import evaluate_answers


def generate_report(file_path: str, output_path: str = None) -> str:
    """
    Run all evaluations and generate a markdown report.

    Args:
        file_path: Path to the test document.
        output_path: Where to save the report (default: docs/evaluation_report.md).

    Returns:
        The report content as a string.
    """
    output_path = output_path or os.path.join(
        os.path.dirname(__file__), "..", "docs", "evaluation_report.md"
    )

    print("📊 Running retrieval evaluation...")
    retrieval_results = run_retrieval_test(file_path)

    print("📝 Running answer quality evaluation...")
    answer_results = evaluate_answers(file_path)

    # Generate report
    report = f"""# 📊 Evaluation Report — Document Q&A Assistant

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Test Document:** {os.path.basename(file_path)}

---

## 1. Retrieval Quality

| Metric | Value |
|--------|-------|
| Total Tests | {retrieval_results['total_tests']} |
| Passed | {retrieval_results['passed']} |
| Failed | {retrieval_results['failed']} |
| **Pass Rate** | **{retrieval_results['pass_rate']:.1%}** |
| **Avg Keyword Recall** | **{retrieval_results['avg_keyword_recall']:.1%}** |

### Per-Query Results

| # | Query | Keywords Found | Recall | Status |
|---|-------|---------------|--------|--------|
"""

    for i, r in enumerate(retrieval_results['details'], 1):
        status = "✅" if r["passed"] else "❌"
        report += f"| {i} | {r['query'][:40]}... | {r['keywords_found']}/{r['keywords_total']} | {r['keyword_recall']:.0%} | {status} |\n"

    report += f"""
---

## 2. Answer Quality (LLM-as-a-Judge)

| Metric | Score (out of 5) |
|--------|-----------------|
| Correctness | {answer_results['averages'].get('correctness', 0):.2f} |
| Groundedness | {answer_results['averages'].get('groundedness', 0):.2f} |
| Relevance | {answer_results['averages'].get('relevance', 0):.2f} |
| Completeness | {answer_results['averages'].get('completeness', 0):.2f} |
| **Overall** | **{answer_results['averages'].get('overall', 0):.2f}** |

### Per-Question Details

"""

    for i, r in enumerate(answer_results['details'], 1):
        s = r['scores']
        report += f"""#### Question {i}: {r['question']}
- **Expected:** {r['expected_answer']}
- **AI Answer:** {r['ai_answer'][:200]}...
- **Scores:** Correctness={s.get('correctness',0)}, Groundedness={s.get('groundedness',0)}, Relevance={s.get('relevance',0)}, Completeness={s.get('completeness',0)}
- **Reasoning:** {s.get('reasoning', 'N/A')}

"""

    report += f"""---

## 3. Performance Summary

| Aspect | Target | Actual | Status |
|--------|--------|--------|--------|
| Retrieval Pass Rate | ≥ 70% | {retrieval_results['pass_rate']:.0%} | {"✅" if retrieval_results['pass_rate'] >= 0.7 else "⚠️"} |
| Answer Correctness | ≥ 4.0 | {answer_results['averages'].get('correctness', 0):.1f} | {"✅" if answer_results['averages'].get('correctness', 0) >= 4.0 else "⚠️"} |
| Answer Groundedness | ≥ 4.5 | {answer_results['averages'].get('groundedness', 0):.1f} | {"✅" if answer_results['averages'].get('groundedness', 0) >= 4.5 else "⚠️"} |
| Overall Quality | ≥ 4.0 | {answer_results['averages'].get('overall', 0):.1f} | {"✅" if answer_results['averages'].get('overall', 0) >= 4.0 else "⚠️"} |

---

## 4. Known Limitations

1. **Single-document scope** — Only one document can be analyzed at a time.
2. **English only** — Non-English documents may produce poor results.
3. **Embedding model** — Using `all-MiniLM-L6-v2` (lightweight); larger models may improve retrieval.
4. **Chunk boundaries** — Information split across chunk boundaries may be missed.
5. **No OCR** — Scanned PDFs (images) are not supported; only text-based PDFs.

## 5. Recommendations

- Increase `TOP_K` for complex queries requiring broader context.
- Use larger embedding models for highly technical or complex documents.
- Fine-tune chunk size based on document structure (e.g., smaller for clauses).
- Consider semantic chunking for better boundary detection.
"""

    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {output_path}")
    return report


if __name__ == "__main__":
    test_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_document.txt")
    if os.path.exists(test_file):
        generate_report(test_file)
    else:
        print(f"Test file not found: {test_file}")
