import json
import pytest
from app.export.exporter import export_as_markdown, export_as_json, export_as_pdf


def test_export_as_markdown():
    report = {
        "question": "What is Agentic RAG?",
        "answer": "Agentic RAG combines reasoning loops and retrieval [S1].",
        "sources": [
            {
                "source_id": "S1",
                "title": "Agentic RAG Paper",
                "url": "https://arxiv.org/abs/example",
                "domain": "arxiv.org",
                "snippet": "We present agentic RAG...",
            }
        ],
        "elapsed_seconds": 4.2,
    }

    md = export_as_markdown(report)
    assert "# Research Report: What is Agentic RAG?" in md
    assert "Agentic RAG Paper" in md
    assert "https://arxiv.org/abs/example" in md
    assert "4.2s" in md


def test_export_as_json():
    report = {
        "question": "What is BGE-M3?",
        "answer": "BGE-M3 supports multi-lingual dense and sparse vectors [S1].",
        "sources": [{"source_id": "S1", "url": "https://example.com"}],
    }

    raw_json = export_as_json(report)
    parsed = json.loads(raw_json)
    assert parsed["question"] == "What is BGE-M3?"
    assert len(parsed["sources"]) == 1
    assert parsed["sources"][0]["source_id"] == "S1"


def test_export_as_pdf():
    report = {
        "question": "What is Hybrid Retrieval?",
        "answer": "Hybrid retrieval fuses dense vector similarity with sparse BM25 scores.",
        "sources": [{"source_id": "S1", "title": "BM25 Guide", "url": "https://example.com"}],
    }

    pdf_bytes = export_as_pdf(report)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 100
    # PDF magic header
    assert pdf_bytes.startswith(b"%PDF")
