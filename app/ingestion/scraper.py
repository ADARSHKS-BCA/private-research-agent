"""
Canonical Scraper re-exports for backward compatibility.
All scraper implementation has been unified into app.web.scraper.
"""

from app.web.scraper import (  # noqa: F401
    ScrapedDocument,
    ScrapedWebDocument,
    Scraper,
    extract_domain,
    generate_document_id,
    get_firecrawl_client,
    scrape_search_results,
    scrape_url,
    scrape_urls,
)

__all__ = [
    "ScrapedDocument",
    "ScrapedWebDocument",
    "Scraper",
    "extract_domain",
    "generate_document_id",
    "get_firecrawl_client",
    "scrape_search_results",
    "scrape_url",
    "scrape_urls",
]
