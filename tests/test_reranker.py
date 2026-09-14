from unittest.mock import MagicMock, patch
import pytest
from app.retrieval.reranker import CrossEncoderReranker, rerank_evidence


def test_reranker_passthrough_when_no_backend():
    cand1 = MagicMock()
    cand1.payload = {"text": "First passage"}
    cand2 = MagicMock()
    cand2.payload = {"text": "Second passage"}

    reranker = CrossEncoderReranker(model_name="mock-model")
    reranker._model = None  # force passthrough

    results = reranker.rerank(query="test query", candidates=[cand1, cand2], top_k=2)
    assert len(results) == 2
    assert results[0] == cand1
    assert results[1] == cand2


def test_rerank_evidence_flag_disabled():
    cand1 = MagicMock()
    cand1.payload = {"text": "Passage A"}
    cand2 = MagicMock()
    cand2.payload = {"text": "Passage B"}

    with patch("app.retrieval.reranker.settings.reranker_enabled", False):
        results = rerank_evidence(query="test query", candidates=[cand1, cand2], top_k=1)
        assert len(results) == 1
        assert results[0] == cand1
