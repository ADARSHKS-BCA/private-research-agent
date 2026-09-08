from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from app.ingestion.pipeline import ingest_url


def test_pipeline_end_to_end(tmp_path: Path):
    sample_markdown = """# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

We present Retrieval-Augmented Generation (RAG) models.

## Abstract
RAG models combine pre-trained parametric and non-parametric memory for language generation.
"""
    with patch("app.ingestion.pipeline.scrape_url") as mock_scrape, \
         patch("app.ingestion.pipeline.get_embedder") as mock_get_embedder, \
         patch("app.ingestion.pipeline.QdrantStore") as mock_store_cls:

        from app.ingestion.scraper import ScrapedDocument
        mock_doc = ScrapedDocument(
            document_id="doc_test_rag",
            url="https://arxiv.org/abs/2005.11401",
            title="RAG Paper",
            markdown=sample_markdown,
            metadata={},
            raw_file_path=str(tmp_path / "raw.md"),
        )
        mock_scrape.return_value = mock_doc

        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [[0.1] * 1024]
        mock_get_embedder.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.upsert_chunks.return_value = 1
        mock_store_cls.return_value = mock_store

        result = ingest_url("https://arxiv.org/abs/2005.11401", collection_name="test_collection")

        assert result["document_id"] == "doc_test_rag"
        assert result["url"] == "https://arxiv.org/abs/2005.11401"
        assert result["chunks_created"] == 1
        assert result["collection"] == "test_collection"
