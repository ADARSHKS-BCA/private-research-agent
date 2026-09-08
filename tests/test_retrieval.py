from unittest.mock import MagicMock, patch
import pytest
from app.retrieval.search import search


def test_retrieval_returns_provenance():
    mock_point = MagicMock()
    mock_point.id = "uuid-123"
    mock_point.score = 0.88
    mock_point.payload = {
        "text": "Agentic RAG methods utilize multi-step reasoning.",
        "document_id": "doc_abc123",
        "chunk_id": "doc_abc123_chunk_0",
        "url": "https://example.com/agentic-rag",
        "title": "Agentic RAG Overview",
        "domain": "example.com",
        "source_type": "web",
        "content_hash": "hash123",
        "created_at": "2026-09-08T00:00:00Z",
    }

    mock_client = MagicMock()
    mock_query_res = MagicMock()
    mock_query_res.points = [mock_point]
    mock_client.query_points.return_value = mock_query_res

    with patch("app.retrieval.search.client", mock_client), \
         patch("app.retrieval.search.get_model") as mock_get_model:

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value.tolist.return_value = [0.1] * 1024
        mock_get_model.return_value = mock_encoder

        results = search("agentic rag", top_k=1)

        assert len(results) == 1
        assert results[0].payload["document_id"] == "doc_abc123"
        assert results[0].payload["url"] == "https://example.com/agentic-rag"
        assert results[0].payload["source_type"] == "web"
        assert results[0].score == 0.88
