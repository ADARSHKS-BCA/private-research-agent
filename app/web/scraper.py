import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.web.search import search_web

# Load environment variables
load_dotenv()


@dataclass
class ScrapedWebDocument:
    url: str
    title: str
    markdown: str
    description: str = ""
    domain: str = ""
    crawled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_domain(url: str) -> str:
    """Extract domain host from URL (e.g. news.microsoft.com)."""
    try:
        parsed = urlparse(url.strip())
        return parsed.netloc.lower()
    except Exception:
        return ""


def get_firecrawl_client(api_key: Optional[str] = None):
    """Get initialized Firecrawl client."""
    key = api_key or os.getenv("FIRECRAWL_API_KEY")
    if not key:
        raise ValueError("FIRECRAWL_API_KEY is not set in .env")

    try:
        from firecrawl import FirecrawlApp
        return FirecrawlApp(api_key=key)
    except (ImportError, AttributeError):
        try:
            from firecrawl import Firecrawl
            return Firecrawl(api_key=key)
        except ImportError:
            raise ImportError("The 'firecrawl-py' package is required. Run: pip install firecrawl-py")


def scrape_with_rest_api(url: str, api_key: str, timeout: int = 30) -> Dict[str, Any]:
    """Direct REST API scrape fallback to ensure maximum reliability."""
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": url,
        "formats": ["markdown"],
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict):
            return data.get("data", {}) or data
    return {}


def scrape_url(
    url: str,
    description: str = "",
    title_hint: str = "",
    timeout: int = 30,
) -> ScrapedWebDocument:
    """
    Scrape a single URL and return a structured ScrapedWebDocument.
    Isolates errors so individual failure does not raise an exception.
    """
    clean_url = url.strip()
    domain = extract_domain(clean_url)
    crawled_at = datetime.now(timezone.utc).isoformat()

    if not clean_url:
        return ScrapedWebDocument(
            url="",
            title="",
            markdown="",
            domain="",
            crawled_at=crawled_at,
            success=False,
            error="Empty URL provided",
        )

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return ScrapedWebDocument(
            url=clean_url,
            title=title_hint or "Untitled",
            markdown="",
            description=description,
            domain=domain,
            crawled_at=crawled_at,
            success=False,
            error="FIRECRAWL_API_KEY is missing in .env",
        )

    raw_markdown = ""
    title = title_hint
    extracted_desc = description
    meta: Dict[str, Any] = {}

    # Strategy 1: Try Python SDK
    try:
        client = get_firecrawl_client(api_key=api_key)
        scrape_res = None

        if hasattr(client, "scrape"):
            try:
                scrape_res = client.scrape(clean_url, formats=["markdown"])
            except TypeError:
                scrape_res = client.scrape(clean_url)
        elif hasattr(client, "scrape_url"):
            scrape_res = client.scrape_url(clean_url, params={"formats": ["markdown"]})

        if scrape_res is not None:
            if hasattr(scrape_res, "model_dump"):
                scrape_res = scrape_res.model_dump()
            elif hasattr(scrape_res, "dict"):
                scrape_res = scrape_res.dict()

            if isinstance(scrape_res, dict):
                raw_markdown = scrape_res.get("markdown", "") or ""
                metadata_dict = scrape_res.get("metadata", {}) or {}
                if isinstance(metadata_dict, dict):
                    title = metadata_dict.get("title", "") or title
                    extracted_desc = metadata_dict.get("description", "") or extracted_desc
                    meta = metadata_dict
                if not title:
                    title = scrape_res.get("title", "") or title
            else:
                raw_markdown = str(getattr(scrape_res, "markdown", None) or "")
                title = str(getattr(scrape_res, "title", None) or title)
                meta_attr = getattr(scrape_res, "metadata", None)
                if meta_attr:
                    meta = meta_attr if isinstance(meta_attr, dict) else getattr(meta_attr, "__dict__", {})
    except Exception as e:
        # SDK error, proceed to REST fallback
        raw_markdown = ""

    # Strategy 2: Direct REST fallback if SDK returned empty markdown
    if not raw_markdown:
        try:
            rest_data = scrape_with_rest_api(clean_url, api_key=api_key, timeout=timeout)
            if rest_data:
                raw_markdown = rest_data.get("markdown", "") or ""
                m = rest_data.get("metadata", {})
                if isinstance(m, dict):
                    title = m.get("title", "") or title
                    extracted_desc = m.get("description", "") or extracted_desc
                    meta = m
        except Exception as e:
            return ScrapedWebDocument(
                url=clean_url,
                title=title or domain,
                markdown="",
                description=description,
                domain=domain,
                crawled_at=crawled_at,
                success=False,
                error=f"Scraping failed: {e}",
            )

    if not raw_markdown:
        return ScrapedWebDocument(
            url=clean_url,
            title=title or domain,
            markdown="",
            description=description,
            domain=domain,
            crawled_at=crawled_at,
            success=False,
            error="No markdown content retrieved from target webpage",
        )

    # Fallback title if missing
    if not title:
        for line in raw_markdown.splitlines():
            s = line.strip()
            if s.startswith("#"):
                title = s.lstrip("#").strip()
                break
        if not title:
            title = domain

    return ScrapedWebDocument(
        url=clean_url,
        title=title,
        markdown=raw_markdown,
        description=extracted_desc,
        domain=domain,
        crawled_at=crawled_at,
        success=True,
        error=None,
        metadata=meta,
    )


def scrape_urls(urls: List[str], timeout: int = 30) -> List[ScrapedWebDocument]:
    """
    Scrape multiple URLs sequentially with error isolation.
    """
    scraped_docs: List[ScrapedWebDocument] = []
    total = len(urls)

    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{total}] Scraping: {url} ...")
        doc = scrape_url(url=url, timeout=timeout)
        if doc.success:
            print(f"       Success: {len(doc.markdown)} chars scraped ('{doc.title}')")
        else:
            print(f"       Failed:  {doc.error}")
        scraped_docs.append(doc)

    return scraped_docs


def scrape_search_results(search_results: List[Dict[str, Any]], timeout: int = 30) -> List[ScrapedWebDocument]:
    """
    Scrape all URLs from a list of search result dictionaries (from Stage 6).
    Preserves original title and description hints from search.
    """
    scraped_docs: List[ScrapedWebDocument] = []
    total = len(search_results)

    for i, item in enumerate(search_results, start=1):
        url = item.get("url", "").strip()
        title_hint = item.get("title", "")
        description_hint = item.get("description", "")

        if not url:
            continue

        print(f"[{i}/{total}] Scraping: {url}")
        doc = scrape_url(
            url=url,
            description=description_hint,
            title_hint=title_hint,
            timeout=timeout,
        )

        if doc.success:
            print(f"       ✓ {len(doc.markdown)} chars ('{doc.title}')")
        else:
            print(f"       ✗ {doc.error}")

        scraped_docs.append(doc)

    return scraped_docs


def main():
    parser = argparse.ArgumentParser(description="Firecrawl Dynamic Multi-URL Scraper (Stage 7)")
    parser.add_argument("--query", type=str, help="Search query to discover and scrape URLs", default=None)
    parser.add_argument("--urls", nargs="+", help="Explicit list of URLs to scrape", default=None)
    parser.add_argument("--limit", type=int, help="Maximum number of URLs to scrape", default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("  Private Research Agent - Dynamic URL Scraper (Stage 7)")
    print("=" * 70)

    urls_to_scrape: List[str] = []
    search_items: List[Dict[str, Any]] = []

    if args.urls:
        urls_to_scrape = args.urls[:args.limit]
    else:
        query = args.query
        if not query:
            query = input("\nWhat research topic do you want to scrape? ").strip()

        if not query:
            print("Empty query. Exiting.")
            return

        print(f"\n[Search] Discovering top {args.limit} URLs for '{query}'...")
        search_items = search_web(query=query, limit=args.limit)

        if not search_items:
            print("No URLs discovered for this query. Exiting.")
            return

        print(f"Found {len(search_items)} URLs. Starting multi-page scraping...\n")

    # Perform scraping
    if search_items:
        docs = scrape_search_results(search_items)
    else:
        docs = scrape_urls(urls_to_scrape)

    # Display results summary
    print("\n" + "=" * 70)
    print("SCRAPING SUMMARY:")
    print("=" * 70)

    successful = [d for d in docs if d.success]
    failed = [d for d in docs if not d.success]

    print(f"Total Scraped: {len(docs)} | Successful: {len(successful)} | Failed: {len(failed)}\n")

    for i, doc in enumerate(docs, start=1):
        status = "SUCCESS" if doc.success else "FAILED"
        print(f"[{i}] [{status}] {doc.title}")
        print(f"    URL:        {doc.url}")
        print(f"    Domain:     {doc.domain}")
        print(f"    Characters: {len(doc.markdown)} chars")
        if doc.error:
            print(f"    Error:      {doc.error}")
        else:
            # Preview first 150 characters of raw markdown
            preview = doc.markdown[:150].replace("\n", " ").strip()
            print(f"    Preview:    {preview}...")
        print()


if __name__ == "__main__":
    main()
