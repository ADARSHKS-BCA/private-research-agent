import pytest
from unittest.mock import MagicMock, patch

from app.agent.graph import create_research_graph, should_continue
from app.agent.nodes import (
    evaluate_evidence,
    generate_answer,
    index_documents,
    plan_research,
    process_documents,
    retrieve_evidence,
    scrape_sources,
    search_web_node,
    validate_citations,
)
from app.agent.runner import run_agent_research
from app.agent.state import ResearchState, create_initial_state
from app.web.scraper import ScrapedWebDocument


# ---------------------------------------------------------------------
# 1. State Creation
# ---------------------------------------------------------------------
def test_state_creation_and_defaults():
    state = create_initial_state("What is LangGraph?", max_iterations=2)
    assert state["question"] == "What is LangGraph?"
    assert state["search_queries"] == []
    assert state["search_results"] == []
    assert state["scraped_documents"] == []
    assert state["cleaned_documents"] == []
    assert state["chunks"] == []
    assert state["retrieved_documents"] == []
    assert state["evidence_sufficient"] is False
    assert state["research_iteration"] == 1
    assert state["max_iterations"] == 2
    assert state["answer"] == ""
    assert state["sources"] == []
    assert state["errors"] == []


# ---------------------------------------------------------------------
# 2. Planner Node Output
# ---------------------------------------------------------------------
def test_plan_research_success():
    state = create_initial_state("Explain agentic RAG workflows")
    with patch("app.agent.nodes._call_llm", return_value='["agentic RAG architectures", "agentic RAG evaluation"]'):
        result = plan_research(state)
        assert "search_queries" in result
        assert len(result["search_queries"]) == 2
        assert result["search_queries"] == ["agentic RAG architectures", "agentic RAG evaluation"]


def test_plan_research_fallback_on_invalid_json():
    state = create_initial_state("What is quantum computing?")
    with patch("app.agent.nodes._call_llm", return_value="Here are your queries: no json here"):
        result = plan_research(state)
        assert len(result["search_queries"]) >= 1
        assert "What is quantum computing?" in result["search_queries"][0]


# ---------------------------------------------------------------------
# 3. Conditional Evidence Routing
# ---------------------------------------------------------------------
def test_should_continue_when_sufficient():
    state = create_initial_state("Query")
    state["evidence_sufficient"] = True
    state["research_iteration"] = 1
    state["max_iterations"] = 3
    assert should_continue(state) == "generate_answer"


def test_should_continue_when_insufficient_and_under_max():
    state = create_initial_state("Query")
    state["evidence_sufficient"] = False
    state["research_iteration"] = 1
    state["max_iterations"] = 3
    assert should_continue(state) == "plan_research"


# ---------------------------------------------------------------------
# 4. Maximum Iteration Limit
# ---------------------------------------------------------------------
def test_should_continue_when_max_iterations_reached():
    state = create_initial_state("Query")
    state["evidence_sufficient"] = False
    state["research_iteration"] = 3
    state["max_iterations"] == 3
    assert should_continue(state) == "generate_answer"


def test_evaluate_evidence_at_max_iterations():
    state = create_initial_state("Query", max_iterations=2)
    state["research_iteration"] = 2
    result = evaluate_evidence(state)
    assert result["evidence_sufficient"] is True


# ---------------------------------------------------------------------
# 5. Citation Validation
# ---------------------------------------------------------------------
def test_validate_citations_in_agent():
    mock_point = MagicMock()
    mock_point.payload = {
        "text": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "title": "LangGraph Documentation",
        "url": "https://langchain-ai.github.io/langgraph/",
        "domain": "langchain-ai.github.io",
        "chunk_id": "doc_1_chunk_0",
        "document_id": "doc_1",
    }
    mock_point.score = 0.95

    state = create_initial_state("What is LangGraph?")
    state["answer"] = "LangGraph is used for building stateful agent workflows [S1], but not [S99]."
    state["retrieved_documents"] = [mock_point]

    result = validate_citations(state)
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_id"] == "S1"
    assert result["sources"][0]["url"] == "https://langchain-ai.github.io/langgraph/"
    assert any("S99" in err for err in result["errors"])


# ---------------------------------------------------------------------
# 6. Empty Search Results Handling
# ---------------------------------------------------------------------
def test_empty_search_results():
    state = create_initial_state("Obscure Query")
    state["search_queries"] = ["Obscure Query"]

    with patch("app.agent.nodes.search_web", return_value=[]):
        result = search_web_node(state)
        assert result["search_results"] == []


# ---------------------------------------------------------------------
# 7. Failed Scraping Handling
# ---------------------------------------------------------------------
def test_failed_scraping_isolation():
    state = create_initial_state("Query")
    state["search_results"] = [
        {"title": "Broken Page", "url": "https://example.com/broken", "description": ""}
    ]

    failed_doc = ScrapedWebDocument(
        url="https://example.com/broken",
        title="Broken Page",
        markdown="",
        domain="example.com",
        success=False,
        error="404 Not Found",
    )

    with patch("app.agent.nodes.scrape_search_results", return_value=[failed_doc]):
        result = scrape_sources(state)
        assert len(result["scraped_documents"]) == 1
        assert result["scraped_documents"][0].success is False


# ---------------------------------------------------------------------
# 8. End-to-End Graph Mock Execution
# ---------------------------------------------------------------------
def test_end_to_end_graph_execution():
    mock_point = MagicMock()
    mock_point.payload = {
        "text": "Agentic RAG combines retrieval mechanisms with autonomous reasoning loops.",
        "title": "Agentic RAG Paper",
        "url": "https://arxiv.org/abs/2401.00000",
        "domain": "arxiv.org",
        "chunk_id": "doc_1_chunk_0",
        "document_id": "doc_1",
    }
    mock_point.score = 0.93

    mock_scraped = [
        ScrapedWebDocument(
            url="https://arxiv.org/abs/2401.00000",
            title="Agentic RAG Paper",
            markdown="# Agentic RAG\n\nAgentic RAG combines retrieval mechanisms with autonomous reasoning loops.",
            domain="arxiv.org",
            success=True,
        )
    ]

    with patch("app.agent.nodes._call_llm") as mock_llm, \
         patch("app.agent.nodes.search_web") as mock_search, \
         patch("app.agent.nodes.scrape_search_results", return_value=mock_scraped), \
         patch("app.agent.nodes.get_embedder") as mock_get_embedder, \
         patch("app.agent.nodes.QdrantStore") as mock_store_cls, \
         patch("app.agent.nodes.search", return_value=[mock_point]):

        # Mock LLM responses in order:
        # 1. plan_research -> queries JSON
        # 2. evaluate_evidence -> sufficient JSON
        # 3. generate_answer -> answer string with citation
        mock_llm.side_effect = [
            '["agentic RAG approaches"]',
            '{"sufficient": true, "reasoning": "Clear explanation found.", "missing_information": ""}',
            "Agentic RAG combines retrieval with autonomous reasoning loops. [S1]",
        ]

        mock_search.return_value = [
            {
                "title": "Agentic RAG Paper",
                "url": "https://arxiv.org/abs/2401.00000",
                "description": "Paper overview",
            }
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [[0.1] * 1024]
        mock_get_embedder.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.upsert_chunks.return_value = 1
        mock_store_cls.return_value = mock_store

        result = run_agent_research("What are agentic RAG approaches?", max_iterations=2)

        assert result["question"] == "What are agentic RAG approaches?"
        assert result["evidence_sufficient"] is True
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source_id"] == "S1"
        assert result["sources"][0]["url"] == "https://arxiv.org/abs/2401.00000"
        assert "Agentic RAG" in result["answer"]
