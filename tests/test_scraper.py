from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from app.ingestion.scraper import Scraper, generate_document_id


def test_generate_document_id_deterministic():
    url1 = "https://arxiv.org/abs/2005.11401"
    url2 = "https://arxiv.org/abs/2005.11401/"
    id1 = generate_document_id(url1)
    id2 = generate_document_id(url2)
    assert id1.startswith("doc_")
    assert id1 == id2


def test_scraper_save_raw_markdown(tmp_path: Path):
    mock_client = MagicMock()
    mock_client.scrape.return_value = {
        "markdown": "# Retrieval-Augmented Generation\n\nPaper details here...",
        "metadata": {"title": "RAG Paper"},
    }

    scraper = Scraper(api_key="test_key", save_dir=tmp_path)
    scraper._client = mock_client

    url = "https://example.com/rag-paper"
    doc = scraper.scrape_url(url)

    assert doc.url == url
    assert doc.title == "RAG Paper"
    assert "Retrieval-Augmented Generation" in doc.markdown
    assert doc.document_id == generate_document_id(url)
    assert doc.raw_file_path is not None

    raw_file = Path(doc.raw_file_path)
    assert raw_file.exists()
    assert raw_file.read_text(encoding="utf-8") == doc.markdown
