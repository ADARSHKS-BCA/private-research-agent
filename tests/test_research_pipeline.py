from unittest.mock import MagicMock, patch
import pytest
from app.research_pipeline import run_research
from app.web.scraper import ScrapedWebDocument


def test_run_research_pipeline_end_to_end():
    mock_search_results = [
        {
            "title": "Agentic RAG Explained",
            "url": "https://example.com/agentic-rag",
            "description": "Overview of agentic RAG and autonomous agents.",
        }
    ]

    mock_scraped_docs = [
        ScrapedWebDocument(
            url="https://example.com/agentic-rag",
            title="Agentic RAG Explained",
            markdown="# Agentic RAG\n\nAgentic RAG systems incorporate autonomous reasoning loops [1].",
            domain="example.com",
            success=True,
        )
    ]

    with patch("app.research_pipeline.search_web", return_value=mock_search_results), \
         patch("app.research_pipeline.scrape_search_results", return_value=mock_scraped_docs), \
         patch("app.research_pipeline.get_embedder") as mock_get_embedder, \
         patch("app.research_pipeline.QdrantStore") as mock_qdrant_store_cls, \
         patch("app.research_pipeline.search") as mock_retrieval, \
         patch("app.research_pipeline.generate_answer_from_retrieved_chunks") as mock_gen_answer:

        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [[0.1] * 1024]
        mock_embedder.get_embedding_dimension.return_value = 1024
        mock_get_embedder.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.upsert_chunks.return_value = 1
        mock_qdrant_store_cls.return_value = mock_store

        mock_point = MagicMock()
        mock_point.score = 0.92
        mock_point.payload = {
            "text": "Agentic RAG systems incorporate autonomous reasoning loops.",
            "title": "Agentic RAG Explained",
            "url": "https://example.com/agentic-rag",
            "domain": "example.com",
            "chunk_id": "doc_1_chunk_0",
            "document_id": "doc_1",
        }
        mock_retrieval.return_value = [mock_point]

        mock_gen_answer.return_value = {
            "query": "What is agentic RAG?",
            "answer": "Agentic RAG incorporates autonomous reasoning loops. [S1]",
            "sources": [
                {
                    "source_id": "S1",
                    "title": "Agentic RAG Explained",
                    "url": "https://example.com/agentic-rag",
                }
            ],
            "invalid_citations": [],
        }

        result = run_research("What is agentic RAG?", num_urls=1, top_k_retrieval=1, stream=False)

        assert result["question"] == "What is agentic RAG?"
        assert result["documents_scraped"] == 1
        assert result["chunks_stored"] == 1
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source_id"] == "S1"
        assert result["sources"][0]["url"] == "https://example.com/agentic-rag"
