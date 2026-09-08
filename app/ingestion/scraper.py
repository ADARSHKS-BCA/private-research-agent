import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    document_id: str
    url: str
    title: str
    markdown: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_document_id(url: str) -> str:
    """
    Generate a deterministic, unique document ID based on normalized URL.
    This guarantees idempotent document references across pipeline runs.
    """
    parsed = urlparse(url.strip())
    # Normalize: lowercase scheme + netloc, strip trailing slash from path
    normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    url_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"doc_{url_hash}"


class Scraper:
    def __init__(self, api_key: Optional[str] = None, save_dir: Optional[Path] = None):
        self.api_key = api_key or settings.firecrawl_api_key
        self.save_dir = Path(save_dir or settings.data_raw_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("FIRECRAWL_API_KEY is not configured. Please set it in .env")
            try:
                # Support firecrawl v1.x / v0.x
                from firecrawl import FirecrawlApp
                self._client = FirecrawlApp(api_key=self.api_key)
            except (ImportError, AttributeError):
                from firecrawl import Firecrawl
                self._client = Firecrawl(api_key=self.api_key)
        return self._client

    def scrape_url(self, url: str) -> ScrapedDocument:
        """
        Scrape a given URL using Firecrawl and save the raw markdown to data/raw/<document_id>.md.
        """
        logger.info(f"Scraping URL: {url}")
        client = self._get_client()

        # Firecrawl scraping
        try:
            # Try scrape method (v1.x or v0.x style)
            if hasattr(client, "scrape"):
                scrape_result = client.scrape(url, formats=["markdown"])
            elif hasattr(client, "scrape_url"):
                scrape_result = client.scrape_url(url, params={"formats": ["markdown"]})
            else:
                raise RuntimeError("Unsupported Firecrawl client interface")
        except Exception as e:
            logger.error(f"Firecrawl failed to scrape {url}: {e}")
            raise RuntimeError(f"Firecrawl scraping error: {e}") from e

        # Extract markdown and metadata
        markdown_content = ""
        title = ""
        meta: Dict[str, Any] = {}

        if isinstance(scrape_result, dict):
            markdown_content = scrape_result.get("markdown", "")
            metadata_dict = scrape_result.get("metadata", {})
            title = metadata_dict.get("title", "") or scrape_result.get("title", "")
            meta = metadata_dict
        else:
            # Object with attributes
            markdown_content = getattr(scrape_result, "markdown", "") or ""
            metadata_attr = getattr(scrape_result, "metadata", None)
            if metadata_attr is not None:
                if isinstance(metadata_attr, dict):
                    title = metadata_attr.get("title", "")
                    meta = metadata_attr
                else:
                    title = getattr(metadata_attr, "title", "") or ""
                    meta = getattr(metadata_attr, "__dict__", {})
            if not title:
                title = getattr(scrape_result, "title", "") or ""

        if not markdown_content:
            raise ValueError(f"No markdown content retrieved from {url}")

        if not title:
            # Fallback: extract first heading or domain
            lines = markdown_content.splitlines()
            for line in lines:
                clean_line = line.strip()
                if clean_line.startswith("#"):
                    title = clean_line.lstrip("#").strip()
                    break
            if not title:
                title = urlparse(url).netloc

        document_id = generate_document_id(url)
        scraped_at = datetime.now(timezone.utc).isoformat()

        doc_metadata = {
            "source_type": "webpage",
            "scraped_at": scraped_at,
            "firecrawl_meta": meta,
        }

        # Save raw markdown file under data/raw/<document_id>.md
        raw_file_path = self.save_dir / f"{document_id}.md"
        raw_file_path.write_text(markdown_content, encoding="utf-8")
        logger.info(f"Saved raw markdown to {raw_file_path}")

        return ScrapedDocument(
            document_id=document_id,
            url=url,
            title=title,
            markdown=markdown_content,
            metadata=doc_metadata,
            raw_file_path=str(raw_file_path),
        )


def scrape_url(url: str) -> ScrapedDocument:
    """Convenience functional interface for scraping."""
    scraper = Scraper()
    return scraper.scrape_url(url)
