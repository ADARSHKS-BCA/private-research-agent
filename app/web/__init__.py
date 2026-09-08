from app.web.scraper import (
    ScrapedWebDocument,
    extract_domain,
    scrape_search_results,
    scrape_url,
    scrape_urls,
)
from app.web.search import search_web

__all__ = [
    "search_web",
    "ScrapedWebDocument",
    "scrape_url",
    "scrape_urls",
    "scrape_search_results",
    "extract_domain",
]
