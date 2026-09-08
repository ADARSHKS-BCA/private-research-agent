from unittest.mock import MagicMock, patch
import pytest
from app.web.search import search_web


def test_search_web_empty_query():
    assert search_web("") == []
    assert search_web("   ") == []


def test_search_web_success():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "data": [
            {
                "title": "Quantum Computing Overview",
                "url": "https://example.com/quantum",
                "description": "An introduction to quantum superposition and entanglement.",
            },
            {
                "title": "Quantum Algorithms",
                "url": "https://example.com/algorithms",
                "description": "Overview of Shor's and Grover's quantum algorithms.",
            },
        ]
    }

    with patch("app.web.search.get_firecrawl_client", return_value=mock_client):
        results = search_web("quantum computing", limit=2)

        assert len(results) == 2
        assert results[0]["title"] == "Quantum Computing Overview"
        assert results[0]["url"] == "https://example.com/quantum"
        assert "superposition" in results[0]["description"]

        assert results[1]["title"] == "Quantum Algorithms"
        assert results[1]["url"] == "https://example.com/algorithms"
