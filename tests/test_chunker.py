import pytest
from app.ingestion.chunker import StructureAwareChunker, chunk_document
from app.ingestion.cleaner import CleanedDocument


def test_chunker_metadata_and_structure():
    doc_id = "doc_test_123"
    url = "https://example.com/ai-paper"
    title = "AI Architecture Paper"

    # Create content with multiple sections
    section1 = "## 1. Introduction\n\n" + "This is section one describing the background. " * 30
    section2 = "\n\n## 2. Methodology\n\n" + "This section outlines the experimental methodology. " * 30
    section3 = "\n\n## 3. Results\n\n" + "Here are the benchmark results across datasets. " * 30

    content = section1 + section2 + section3

    doc = CleanedDocument(
        document_id=doc_id,
        url=url,
        title=title,
        markdown=content,
        metadata={"source_type": "research_paper"},
    )

    # Use smaller chunk_size for testing multiple splits
    chunker = StructureAwareChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1

    for idx, chunk in enumerate(chunks):
        assert chunk.document_id == doc_id
        assert chunk.chunk_id == f"{doc_id}_chunk_{idx}"
        assert chunk.chunk_index == idx
        assert chunk.total_chunks == len(chunks)
        assert chunk.url == url
        assert chunk.title == title
        assert chunk.source_type == "research_paper"
        assert chunk.created_at is not None
        assert len(chunk.text) > 0


def test_chunker_overlap_validation():
    with pytest.raises(ValueError):
        StructureAwareChunker(chunk_size=100, chunk_overlap=100)
