import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.api.server import app
from app.agent.state import create_initial_state


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "llm_provider" in data


def test_stream_research_empty_query(client):
    response = client.post("/api/research/stream", json={"question": ""})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_stream_adapter_safe_events():
    from app.api.adapter import stream_research_events

    mock_graph = MagicMock()
    # Mock LangGraph yielding node updates
    mock_graph.stream.return_value = [
        {"plan_research": {"search_queries": ["query 1", "query 2"]}},
        {"search_web": {"search_results": [{"title": "Source 1", "url": "https://example.com/1"}]}},
        {"generate_answer": {"answer": "Agentic RAG combines reasoning loops [S1]."}},
        {"validate_citations": {"sources": [{"source_id": "S1", "title": "Source 1", "url": "https://example.com/1"}]}},
    ]

    events = []
    with patch("app.api.adapter.create_research_graph", return_value=mock_graph):
        async for event_chunk in stream_research_events("What is agentic RAG?"):
            events.append(event_chunk)

    joined = "".join(events)
    assert "event: status" in joined
    assert "event: token" in joined
    assert "event: sources" in joined
    assert "event: done" in joined
    # Verify no private thoughts or system prompts are in the status payload
    assert "system_prompt" not in joined
    assert "reasoning" not in joined
