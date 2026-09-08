from unittest.mock import MagicMock
import pytest
from app.rag.citations import (
    Citation,
    extract_citations,
    format_final_sources_block,
    format_prompt_context_with_sources,
    validate_citations,
)


def test_extract_citations_and_format_prompt():
    mock_res_1 = MagicMock()
    mock_res_1.payload = {
        "title": "Agentic RAG Paper",
        "url": "https://arxiv.org/abs/2407.21059",
        "domain": "arxiv.org",
        "chunk_id": "doc_1_chunk_0",
        "document_id": "doc_1",
        "text": "Agentic RAG models use autonomous routing and evaluation loops.",
    }
    mock_res_1.score = 0.8543

    mock_res_2 = MagicMock()
    mock_res_2.payload = {
        "title": "Modular RAG Survey",
        "url": "https://arxiv.org/abs/2312.10997",
        "domain": "arxiv.org",
        "chunk_id": "doc_2_chunk_0",
        "document_id": "doc_2",
        "text": "Modular RAG introduces customizable retriever and generator blocks.",
    }
    mock_res_2.score = 0.7912

    citations = extract_citations([mock_res_1, mock_res_2])

    assert len(citations) == 2
    assert citations[0].source_id == "S1"
    assert citations[0].title == "Agentic RAG Paper"
    assert citations[1].source_id == "S2"

    prompt_context = format_prompt_context_with_sources(citations)
    assert "[S1]" in prompt_context
    assert "[S2]" in prompt_context
    assert "Agentic RAG Paper" in prompt_context


def test_validate_citations_valid_and_invalid():
    c1 = Citation(
        source_id="S1",
        source_index=1,
        title="Doc 1",
        url="https://example.com/1",
        domain="example.com",
        chunk_id="c1",
        document_id="d1",
        score=0.9,
        snippet="Text 1",
    )
    c2 = Citation(
        source_id="S2",
        source_index=2,
        title="Doc 2",
        url="https://example.com/2",
        domain="example.com",
        chunk_id="c2",
        document_id="d2",
        score=0.8,
        snippet="Text 2",
    )

    answer_with_citations = (
        "Agentic RAG introduces autonomous routing [S1]. "
        "It also supports modular workflows [S2], but non-existent claims like [S99] should be caught."
    )

    valid_cites, invalid_ids = validate_citations(answer_with_citations, [c1, c2])

    assert len(valid_cites) == 2
    assert [c.source_id for c in valid_cites] == ["S1", "S2"]
    assert "S99" in invalid_ids


def test_validate_citations_refusal():
    c1 = Citation(
        source_id="S1",
        source_index=1,
        title="Doc 1",
        url="https://example.com/1",
        domain="example.com",
        chunk_id="c1",
        document_id="d1",
        score=0.9,
        snippet="Text 1",
    )

    refusal = "I don't have enough information in the provided sources to answer this question."
    valid_cites, invalid_ids = validate_citations(refusal, [c1])

    assert len(valid_cites) == 0
    assert len(invalid_ids) == 0


def test_format_final_sources_block():
    c1 = Citation(
        source_id="S1",
        source_index=1,
        title="Paper Title",
        url="https://arxiv.org/abs/2005.11401",
        domain="arxiv.org",
        chunk_id="c1",
        document_id="d1",
        score=0.9,
        snippet="Snippet",
    )
    formatted = format_final_sources_block([c1])
    assert "Sources:" in formatted
    assert "[S1] Paper Title" in formatted
    assert "https://arxiv.org/abs/2005.11401" in formatted
