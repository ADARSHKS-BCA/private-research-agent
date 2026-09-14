"""
Automated Evaluation Suite for Private Research Agent.

Evaluates:
- Context Precision: Proportion of retrieved passages relevant to question
- Context Recall: Coverage of ground-truth key concepts in retrieved evidence
- Faithfulness: Groundedness of generated statements with respect to sources
- Citation Accuracy: Precision of citation tags mapped to actual sources
- Latency & Iterations: Total runtime and search loop count

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --dataset evaluation/dataset.json --mock
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.runner import run_agent_research

logger = logging.getLogger(__name__)


def compute_context_recall(key_concepts: List[str], retrieved_texts: List[str]) -> float:
    """
    Compute recall of key ground-truth concepts across all retrieved chunks.
    """
    if not key_concepts:
        return 1.0
    combined_text = " ".join(retrieved_texts).lower()
    matches = sum(1 for concept in key_concepts if concept.lower() in combined_text)
    return round(matches / len(key_concepts), 4)


def compute_context_precision(key_concepts: List[str], retrieved_texts: List[str]) -> float:
    """
    Compute precision: fraction of retrieved chunks that contain at least one key concept.
    """
    if not retrieved_texts:
        return 0.0
    relevant_chunks = 0
    for text in retrieved_texts:
        lower = text.lower()
        if any(concept.lower() in lower for concept in key_concepts):
            relevant_chunks += 1
    return round(relevant_chunks / len(retrieved_texts), 4)


def compute_citation_accuracy(answer: str, sources: List[Dict[str, Any]]) -> float:
    """
    Measure citation accuracy: percentage of [S#] citations in the answer
    that correspond to actual verified source IDs.
    """
    cited_ids = set(re.findall(r"\[(S\d+)\]", answer))
    if not cited_ids:
        # If no citations used, check if sources were present
        return 1.0 if not sources else 0.5

    valid_ids = {s.get("source_id") for s in sources if s.get("source_id")}
    accurate_citations = sum(1 for cid in cited_ids if cid in valid_ids)
    return round(accurate_citations / len(cited_ids), 4)


def compute_faithfulness(answer: str, retrieved_texts: List[str]) -> float:
    """
    Estimate answer faithfulness: fraction of sentences in answer
    supported by at least 2 consecutive content words from retrieved context.
    """
    if not answer or not retrieved_texts:
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.\n]", answer) if len(s.strip()) > 15]
    if not sentences:
        return 1.0

    combined_context = " ".join(retrieved_texts).lower()
    supported = 0

    for sent in sentences:
        words = re.findall(r"\b[a-z]{4,}\b", sent.lower())
        if len(words) < 2:
            supported += 1
            continue
        # Check bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        matched_bigrams = sum(1 for bg in bigrams if bg in combined_context)
        if matched_bigrams > 0:
            supported += 1

    return round(supported / len(sentences), 4)


def run_evaluation(
    dataset_path: Path,
    mock_mode: bool = False,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run evaluation loop over benchmark questions.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 75)
    print("  PRIVATE RESEARCH AGENT - BENCHMARK EVALUATION SUITE")
    print("=" * 75)
    print(f"Dataset: {dataset_path} ({len(cases)} test cases)")
    print(f"Mode: {'MOCK SIMULATION' if mock_mode else 'LIVE LANGGRAPH AGENT'}\n")

    for i, item in enumerate(cases, start=1):
        qid = item.get("id", f"case_{i}")
        question = item["question"]
        key_concepts = item.get("key_concepts", [])

        print(f"[{i}/{len(cases)}] Evaluating: \"{question[:60]}...\"")

        start_time = time.time()

        if mock_mode:
            # Deterministic simulation for tests/CI
            retrieved_texts = [
                f"Sample passage discussing {concept} with detailed technical context."
                for concept in key_concepts[:4]
            ]
            answer = f"According to [S1], {key_concepts[0] if key_concepts else 'the system'} operates effectively."
            sources = [{"source_id": "S1", "url": "https://arxiv.org/abs/example", "title": "Example Paper"}]
            iterations = 1
            elapsed = 0.05
        else:
            try:
                run_res = run_agent_research(question=question, max_iterations=3)
                answer = run_res.get("answer", "")
                sources = run_res.get("sources", [])
                iterations = run_res.get("research_iteration", 1)
                elapsed = run_res.get("elapsed_seconds", round(time.time() - start_time, 2))
                # Retrieve texts from snippets or chunks
                retrieved_texts = [
                    s.get("snippet", "") for s in sources if s.get("snippet")
                ]
                if not retrieved_texts:
                    retrieved_texts = [answer]
            except Exception as e:
                logger.error(f"Error evaluating '{question}': {e}")
                answer = ""
                sources = []
                iterations = 1
                elapsed = round(time.time() - start_time, 2)
                retrieved_texts = []

        # Calculate metrics
        c_recall = compute_context_recall(key_concepts, retrieved_texts)
        c_precision = compute_context_precision(key_concepts, retrieved_texts)
        cit_acc = compute_citation_accuracy(answer, sources)
        faith = compute_faithfulness(answer, retrieved_texts)

        results.append({
            "id": qid,
            "question": question,
            "latency_seconds": elapsed,
            "iterations": iterations,
            "context_recall": c_recall,
            "context_precision": c_precision,
            "citation_accuracy": cit_acc,
            "faithfulness": faith,
            "sources_count": len(sources),
        })

    # Summary aggregations
    avg_recall = round(sum(r["context_recall"] for r in results) / len(results), 4)
    avg_precision = round(sum(r["context_precision"] for r in results) / len(results), 4)
    avg_citation_acc = round(sum(r["citation_accuracy"] for r in results) / len(results), 4)
    avg_faithfulness = round(sum(r["faithfulness"] for r in results) / len(results), 4)
    avg_latency = round(sum(r["latency_seconds"] for r in results) / len(results), 2)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "mode": "mock" if mock_mode else "live",
        "metrics": {
            "avg_context_recall": avg_recall,
            "avg_context_precision": avg_precision,
            "avg_citation_accuracy": avg_citation_acc,
            "avg_faithfulness": avg_faithfulness,
            "avg_latency_seconds": avg_latency,
        },
        "cases": results,
    }

    print("\n" + "=" * 75)
    print("  EVALUATION SUMMARY")
    print("=" * 75)
    print(f"  Context Recall:      {avg_recall * 100:.1f}%")
    print(f"  Context Precision:   {avg_precision * 100:.1f}%")
    print(f"  Citation Accuracy:   {avg_citation_acc * 100:.1f}%")
    print(f"  Faithfulness:        {avg_faithfulness * 100:.1f}%")
    print(f"  Average Latency:     {avg_latency:.2f}s")
    print("=" * 75 + "\n")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Detailed evaluation results saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Private Research Agent performance")
    parser.add_argument("--dataset", type=str, default="evaluation/dataset.json", help="Path to benchmark JSON")
    parser.add_argument("--output", type=str, default="evaluation/results.json", help="Output path for eval report")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without invoking LLMs/Qdrant")
    args = parser.parse_args()

    dataset_file = Path(args.dataset)
    output_file = Path(args.output) if args.output else None
    run_evaluation(dataset_path=dataset_file, mock_mode=args.mock, output_path=output_file)


if __name__ == "__main__":
    main()
