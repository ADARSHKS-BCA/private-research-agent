import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.ingestion.doc_parser import (
    generate_file_document_id,
    parse_text,
    parse_document,
    ingest_document_file,
)


def test_generate_file_document_id():
    doc_id1 = generate_file_document_id("sample.pdf", b"hello world")
    doc_id2 = generate_file_document_id("sample.pdf", b"hello world")
    doc_id3 = generate_file_document_id("sample.pdf", b"different content")

    assert doc_id1 == doc_id2
    assert doc_id1 != doc_id3
    assert doc_id1.startswith("file_sample_")


def test_parse_text_and_markdown():
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = Path(tmpdir) / "notes.txt"
        txt_path.write_text("Line 1 of technical notes.\nLine 2 of technical notes.", encoding="utf-8")

        parsed = parse_text(txt_path)
        assert parsed.filename == "notes.txt"
        assert parsed.document_type == "text"
        assert "Line 1" in parsed.markdown

        md_path = Path(tmpdir) / "paper.md"
        md_path.write_text("# Research Document\n\n## Abstract\nOverview of RAG.", encoding="utf-8")

        parsed_md = parse_document(md_path)
        assert parsed_md.filename == "paper.md"
        assert parsed_md.title == "Research Document"
        assert parsed_md.document_type == "markdown"


def test_ingest_document_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = Path(tmpdir) / "notes.txt"
        txt_path.write_text("Agentic RAG methods utilize multi-step planning and reflection.", encoding="utf-8")

        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.return_value = [[0.1] * 1024]

        mock_store = MagicMock()
        mock_store.upsert_chunks.return_value = 1

        with patch("app.ingestion.doc_parser.get_embedder", return_value=mock_embedder), \
             patch("app.ingestion.doc_parser.QdrantStore", return_value=mock_store):

            result = ingest_document_file(txt_path)
            assert result["status"] == "success"
            assert result["filename"] == "notes.txt"
            assert result["chunks_indexed"] == 1
            mock_store.upsert_chunks.assert_called_once()
