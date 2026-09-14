from unittest.mock import MagicMock, patch
import pytest
from app.retrieval.hybrid import BM25Scorer, HybridRetriever, hybrid_search, tokenize


def test_tokenize():
    tokens = tokenize("Agentic RAG with BGE-M3 embeddings & BM25!")
    assert "agentic" in tokens
    assert "rag" in tokens
    assert "bge-m3" in tokens
    assert "embeddings" in tokens
    assert "bm25" in tokens


def test_bm25_scorer():
    corpus = [
        "Dense retrieval uses neural vectors for semantic similarity.",
        "BM25 is a term-matching ranking function used in information retrieval.",
        "Hybrid search combines dense embeddings and BM25 lexical scores.",
    ]
    scorer = BM25Scorer()
    scorer.fit(corpus)
    assert scorer.corpus_size == 3

    scores_dense = scorer.score("dense vectors", corpus)
    assert len(scores_dense) == 3
    # First document should have highest score for "dense vectors"
    assert scores_dense[0] > scores_dense[1]

    scores_bm25 = scorer.score("information retrieval", corpus)
    # Second document should have highest score
    assert scores_bm25[1] > scores_bm25[0]


def test_hybrid_retrieval_fusion():
    mock_point1 = MagicMock()
    mock_point1.id = "p1"
    mock_point1.score = 0.90
    mock_point1.payload = {"text": "Vector search with deep learning models."}

    mock_point2 = MagicMock()
    mock_point2.id = "p2"
    mock_point2.score = 0.50
    mock_point2.payload = {"text": "BM25 keyword matching for exact acronyms."}

    with patch("app.retrieval.hybrid.dense_search") as mock_dense:
        mock_dense.return_value = [mock_point1, mock_point2]

        retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3)
        results = retriever.retrieve("keyword matching", top_k=2)

        assert len(results) == 2
        # Both results should have hybrid_score, dense_score, and bm25_score attributes
        assert hasattr(results[0], "hybrid_score")
        assert hasattr(results[0], "bm25_score")
        assert hasattr(results[0], "dense_score")
