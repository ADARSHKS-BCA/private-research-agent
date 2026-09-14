import argparse
import concurrent.futures
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.web.search import search_web

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ScrapedWebDocument:
    url: str
    title: str
    markdown: str
    document_id: str = ""
    description: str = ""
    domain: str = ""
    crawled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success: bool = True
    error: Optional[str] = None
    raw_file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias for backward compatibility with ingestion.scraper
ScrapedDocument = ScrapedWebDocument


def generate_document_id(url: str) -> str:
    """
    Generate a deterministic, unique document ID based on normalized URL.
    This guarantees idempotent document references across pipeline runs.
    """
    parsed = urlparse(url.strip())
    normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    url_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"doc_{url_hash}"


def extract_domain(url: str) -> str:
    """Extract domain host from URL (e.g. news.microsoft.com)."""
    try:
        parsed = urlparse(url.strip())
        return parsed.netloc.lower()
    except Exception:
        return ""


def get_firecrawl_client(api_key: Optional[str] = None):
    """Get initialized Firecrawl client."""
    key = api_key or settings.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
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
    elif resp.status_code == 429:
        raise requests.exceptions.HTTPError(f"HTTP 429 Too Many Requests from Firecrawl API")
    return {}


def scrape_url(
    url: str,
    description: str = "",
    title_hint: str = "",
    timeout: Optional[int] = None,
    save_dir: Optional[Path] = None,
    client_override: Optional[Any] = None,
    max_retries: Optional[int] = None,
) -> ScrapedWebDocument:
    """
    Scrape a single URL with retry and exponential backoff.
    Returns a structured ScrapedWebDocument.
    Isolates errors so individual failure does not raise an unhandled exception.
    """
    clean_url = url.strip()
    domain = extract_domain(clean_url)
    crawled_at = datetime.now(timezone.utc).isoformat()
    req_timeout = timeout or settings.request_timeout
    retries = max_retries if max_retries is not None else settings.max_retries

    if not clean_url:
        return ScrapedWebDocument(
            url="",
            title="",
            markdown="",
            document_id="",
            domain="",
            crawled_at=crawled_at,
            success=False,
            error="Empty URL provided",
        )

    doc_id = generate_document_id(clean_url)
    api_key = settings.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY", "")

    if not api_key and client_override is None:
        return ScrapedWebDocument(
            url=clean_url,
            title=title_hint or "Untitled",
            markdown="",
            document_id=doc_id,
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
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 2):
        try:
            # Strategy 1: Try Firecrawl Client (either injected/mocked or standard)
            client = client_override if client_override is not None else get_firecrawl_client(api_key=api_key)
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

            # Strategy 2: Direct REST fallback if client gave empty markdown and no client override
            if not raw_markdown and client_override is None and api_key:
                rest_data = scrape_with_rest_api(clean_url, api_key=api_key, timeout=req_timeout)
                if rest_data:
                    raw_markdown = rest_data.get("markdown", "") or ""
                    m = rest_data.get("metadata", {})
                    if isinstance(m, dict):
                        title = m.get("title", "") or title
                        extracted_desc = m.get("description", "") or extracted_desc
                        meta = m

            if raw_markdown:
                break

        except Exception as e:
            last_err = e
            if attempt <= retries:
                backoff_wait = 1.5 ** attempt
                logger.warning(f"Scrape retry {attempt}/{retries} for {clean_url} after {backoff_wait:.1f}s: {e}")
                time.sleep(backoff_wait)
                continue
            break

    if not raw_markdown:
        err_msg = str(last_err) if last_err else "No markdown content retrieved from target webpage"
        return ScrapedWebDocument(
            url=clean_url,
            title=title or domain,
            markdown="",
            document_id=doc_id,
            description=description,
            domain=domain,
            crawled_at=crawled_at,
            success=False,
            error=err_msg,
        )

    # Fallback title if missing
    if not title:
        for line in raw_markdown.splitlines():
            s = line.strip()
            if s.startswith("#"):
                title = s.lstrip("#").strip()
                break
        if not title:
            title = domain or "Untitled Web Source"

    # Save raw markdown file under data/raw/<document_id>.md
    target_dir = Path(save_dir or settings.data_raw_dir)
    raw_file_path = None
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / f"{doc_id}.md"
        out_file.write_text(raw_markdown, encoding="utf-8")
        raw_file_path = str(out_file)
    except Exception as e:
        logger.warning(f"Could not persist raw markdown for {doc_id}: {e}")

    return ScrapedWebDocument(
        url=clean_url,
        title=title,
        markdown=raw_markdown,
        document_id=doc_id,
        description=extracted_desc,
        domain=domain,
        crawled_at=crawled_at,
        success=True,
        error=None,
        raw_file_path=raw_file_path,
        metadata=meta,
    )


def scrape_urls(
    urls: List[str],
    timeout: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[ScrapedWebDocument]:
    """
    Scrape multiple URLs concurrently with error isolation, maintaining original input order.
    """
    if not urls:
        return []

    workers = max_workers or settings.max_concurrent_scrapes or 5
    workers = max(1, min(workers, len(urls)))
    req_timeout = timeout or settings.request_timeout

    results: List[Optional[ScrapedWebDocument]] = [None] * len(urls)

    def _worker(idx_url_tuple):
        idx, target_url = idx_url_tuple
        try:
            return idx, scrape_url(url=target_url, timeout=req_timeout)
        except Exception as ex:
            return idx, ScrapedWebDocument(
                url=target_url,
                title="Untitled",
                markdown="",
                domain=extract_domain(target_url),
                success=False,
                error=str(ex),
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, (i, url)) for i, url in enumerate(urls)]
        for future in concurrent.futures.as_completed(futures):
            idx, doc = future.result()
            results[idx] = doc

    return [d for d in results if d is not None]


def scrape_search_results(
    search_results: List[Dict[str, Any]],
    timeout: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[ScrapedWebDocument]:
    """
    Scrape all URLs from search result dictionaries concurrently, preserving original ordering.
    """
    if not search_results:
        return []

    workers = max_workers or settings.max_concurrent_scrapes or 5
    workers = max(1, min(workers, len(search_results)))
    req_timeout = timeout or settings.request_timeout

    results: List[Optional[ScrapedWebDocument]] = [None] * len(search_results)

    def _worker(item_tuple):
        idx, item = item_tuple
        url = item.get("url", "").strip() if isinstance(item, dict) else ""
        title_hint = item.get("title", "") if isinstance(item, dict) else ""
        desc_hint = item.get("description", "") if isinstance(item, dict) else ""
        if not url:
            return idx, None
        try:
            doc = scrape_url(
                url=url,
                description=desc_hint,
                title_hint=title_hint,
                timeout=req_timeout,
            )
            return idx, doc
        except Exception as ex:
            return idx, ScrapedWebDocument(
                url=url,
                title=title_hint or "Untitled",
                markdown="",
                domain=extract_domain(url),
                success=False,
                error=str(ex),
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, (i, item)) for i, item in enumerate(search_results)]
        for future in concurrent.futures.as_completed(futures):
            idx, doc = future.result()
            results[idx] = doc

    return [d for d in results if d is not None]


class Scraper:
    """
    Canonical Scraper class interface for pipeline integration and backward compatibility.
    """
    def __init__(self, api_key: Optional[str] = None, save_dir: Optional[Path] = None):
        self.api_key = api_key or settings.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self.save_dir = Path(save_dir or settings.data_raw_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def scrape_url(self, url: str) -> ScrapedWebDocument:
        """Scrape a URL and save raw markdown to disk."""
        doc = scrape_url(
            url=url,
            save_dir=self.save_dir,
            client_override=self._client,
        )
        if not doc.success and doc.error:
            # Maintain backward compatibility with tests expecting RuntimeError on failure
            raise RuntimeError(f"Scraping failed: {doc.error}")
        return doc


def main():
    parser = argparse.ArgumentParser(description="Canonical Multi-URL Web Scraper with Concurrency")
    parser.add_argument("--query", type=str, help="Search query to discover and scrape URLs", default=None)
    parser.add_argument("--urls", nargs="+", help="Explicit list of URLs to scrape", default=None)
    parser.add_argument("--limit", type=int, help="Maximum number of URLs to scrape", default=4)
    args = parser.parse_args()

    print("=" * 70)
    print("  Private Research Agent - Parallel Web Scraper")
    print("=" * 70)

    if args.urls:
        urls = args.urls[:args.limit]
        print(f"Scraping {len(urls)} URLs in parallel...")
        docs = scrape_urls(urls)
    else:
        query = args.query or input("\nWhat research topic do you want to scrape? ").strip()
        if not query:
            print("Empty query. Exiting.")
            return
        print(f"\n[Search] Discovering URLs for '{query}'...")
        search_items = search_web(query=query, limit=args.limit)
        print(f"Discovered {len(search_items)} URLs. Scraping in parallel...")
        docs = scrape_search_results(search_items)

    successful = [d for d in docs if d.success]
    print(f"\nScraped {len(successful)}/{len(docs)} URLs successfully.")
    for idx, doc in enumerate(docs, start=1):
        status = "SUCCESS" if doc.success else "FAILED"
        print(f"[{idx}] [{status}] {doc.title} ({doc.url})")


if __name__ == "__main__":
    main()
