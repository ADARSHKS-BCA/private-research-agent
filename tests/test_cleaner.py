from pathlib import Path
import pytest
from app.processing.cleaner import (
    calculate_content_hash,
    clean_markdown,
    generate_document_id,
    process_scraped_documents,
)
from app.web.scraper import ScrapedWebDocument


def test_generate_document_id_deterministic():
    url1 = "https://example.com/research/paper1"
    url2 = "https://example.com/research/paper1/"
    id1 = generate_document_id(url1)
    id2 = generate_document_id(url2)
    assert id1.startswith("doc_")
    assert id1 == id2


def test_calculate_content_hash():
    text1 = "This is a clean research article."
    text2 = "This   is  a clean   research article.  "
    assert calculate_content_hash(text1) == calculate_content_hash(text2)


def test_clean_markdown():
    raw = """
[Skip to main article](https://example.com)
[Accept cookies](https://example.com/cookies)

# Research Title

This is substantive paragraph content.

* Item 1
* Item 2

```python
def test():
    pass
```

[Share on Twitter](https://twitter.com)
"""
    cleaned = clean_markdown(raw)
    assert "Skip to main article" not in cleaned
    assert "Accept cookies" not in cleaned
    assert "Share on Twitter" not in cleaned
    assert "# Research Title" in cleaned
    assert "substantive paragraph content." in cleaned
    assert "def test():" in cleaned


def test_process_scraped_documents_duplicate_filtering(tmp_path: Path):
    doc1 = ScrapedWebDocument(
        url="https://example.com/page1",
        title="Page 1",
        markdown="# Content One\n\nSubstantive details here.",
        domain="example.com",
        success=True,
    )
    # Duplicate URL
    doc2 = ScrapedWebDocument(
        url="https://example.com/page1",
        title="Page 1 Duplicate",
        markdown="# Content One\n\nSubstantive details here.",
        domain="example.com",
        success=True,
    )
    # Duplicate content hash but different URL
    doc3 = ScrapedWebDocument(
        url="https://example.com/page3",
        title="Page 3",
        markdown="# Content One\n\nSubstantive details here.",
        domain="example.com",
        success=True,
    )
    # Unique doc
    doc4 = ScrapedWebDocument(
        url="https://example.com/page4",
        title="Page 4",
        markdown="# Content Unique\n\nCompletely different research.",
        domain="example.com",
        success=True,
    )

    processed = process_scraped_documents(
        [doc1, doc2, doc3, doc4],
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    # doc1 and doc4 should be kept; doc2 and doc3 are duplicates
    assert len(processed) == 2
    assert processed[0].url == "https://example.com/page1"
    assert processed[1].url == "https://example.com/page4"
