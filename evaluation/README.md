# Private Research Agent Evaluation Suite

This evaluation suite measures the quantitative accuracy, groundedness, and retrieval performance of the Private Research Agent across curated benchmark questions.

## Metrics Evaluated

| Metric | Target | Description |
| :--- | :--- | :--- |
| **Context Recall** | $\ge 80\%$ | Measures whether ground-truth key technical concepts and facts appear in the retrieved evidence passages. |
| **Context Precision** | $\ge 70\%$ | Measures the proportion of retrieved chunks that contain relevant concepts rather than noise. |
| **Faithfulness** | $\ge 85\%$ | Quantifies groundedness: verifies that generated claims are directly supported by retrieved evidence text rather than hallucinated. |
| **Citation Accuracy** | $100\%$ | Verifies that all `[S1]`, `[S2]` citation tags map to valid, non-fabricated retrieved sources. |
| **Latency** | $< 15\text{s}$ | Average total execution time from query receipt to citation validation. |
| **Iterations** | $1 - 3$ | Number of autonomous search loops executed by the LangGraph agent. |

---

## Running the Benchmark

### 1. Offline / Mock Mode (CI & Unit Testing)
Runs without network calls, LLM API keys, or Qdrant:
```bash
python -m evaluation.run_eval --dataset evaluation/dataset.json --mock --output evaluation/results.json
```

### 2. Live Agent Benchmark
Runs the full LangGraph research agent against live web search and vector retrieval:
```bash
python -m evaluation.run_eval --dataset evaluation/dataset.json --output evaluation/results.json
```

---

## Dataset Schema (`evaluation/dataset.json`)

Each benchmark test case contains:
```json
{
  "id": "eval_01",
  "question": "What is the difference between dense retrieval and sparse BM25 retrieval in RAG?",
  "ground_truth": "Dense retrieval uses continuous semantic embeddings...",
  "key_concepts": ["dense retrieval", "sparse", "bm25", "hybrid", "fusion"],
  "expected_domains": ["arxiv.org", "qdrant.tech"]
}
```
