from unittest.mock import MagicMock, patch
import pytest
from app.web.scraper import (
    ScrapedWebDocument,
    extract_domain,
    scrape_search_results,
    scrape_url,
    scrape_urls,
)


def test_extract_domain():
    assert extract_domain("https://news.microsoft.com/source/ai") == "news.microsoft.com"
    assert extract_domain("https://arxiv.org/abs/2005.11401") == "arxiv.org"
    assert extract_domain("http://example.com:8080/path") == "example.com:8080"


def test_scrape_url_success():
    mock_client = MagicMock()
    mock_client.scrape.return_value = {
        "markdown": "# Research on Agentic RAG\n\nAgentic RAG models use autonomous routing...",
        "metadata": {
            "title": "Agentic RAG Paper",
            "description": "A comprehensive study on autonomous RAG systems.",
        },
    }

    with patch("app.web.scraper.get_firecrawl_client", return_value=mock_client):
        doc = scrape_url("https://arxiv.org/abs/2401.00001", title_hint="Hint Title")

        assert doc.success is True
        assert doc.url == "https://arxiv.org/abs/2401.00001"
        assert doc.domain == "arxiv.org"
        assert doc.title == "Agentic RAG Paper"
        assert "Agentic RAG models" in doc.markdown
        assert doc.error is None
        assert doc.crawled_at is not None


def test_scrape_urls_isolates_failures():
    def mock_scrape_side_effect(url, **kwargs):
        if "bad-site" in url:
            return ScrapedWebDocument(
                url=url,
                title="Bad Site",
                markdown="",
                domain="bad-site.com",
                success=False,
                error="HTTP 404 Not Found",
            )
        return ScrapedWebDocument(
            url=url,
            title="Good Site",
            markdown="# Good Content",
            domain="good-site.com",
            success=True,
            error=None,
        )

    with patch("app.web.scraper.scrape_url", side_effect=mock_scrape_side_effect):
        results = scrape_urls(["https://good-site.com", "https://bad-site.com"])

        assert len(results) == 2
        assert results[0].success is True
        assert results[0].title == "Good Site"

        assert results[1].success is False
        assert "404" in results[1].error


def test_scrape_search_results():
    search_data = [
        {
            "title": "Search Title 1",
            "url": "https://example.com/page1",
            "description": "Description 1",
        }
    ]

    with patch("app.web.scraper.scrape_url") as mock_scrape:
        mock_scrape.return_value = ScrapedWebDocument(
            url="https://example.com/page1",
            title="Scraped Title 1",
            markdown="# Page 1 Content",
            domain="example.com",
            success=True,
        )

        docs = scrape_search_results(search_data)
        assert len(docs) == 1
        assert docs[0].url == "https://example.com/page1"
        assert docs[0].success is True
