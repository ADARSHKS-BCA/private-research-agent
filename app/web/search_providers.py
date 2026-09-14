"""
Search Provider Abstraction with automated fallback.

Primary: Firecrawl (deep structured search)
Fallback: DuckDuckGo (free, zero API key required, highly reliable)
"""

from abc import ABC, abstractmethod
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse
import requests

from app.config import settings

logger = logging.getLogger(__name__)


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for query and return list of dicts with title, url, description."""
        pass


class FirecrawlSearchProvider(SearchProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY", "")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY is not configured.")

        # Try Firecrawl SDK first
        try:
            from app.web.scraper import get_firecrawl_client
            client = get_firecrawl_client(api_key=self.api_key)

            raw_response = None
            if hasattr(client, "search"):
                try:
                    raw_response = client.search(query, limit=limit)
                except TypeError:
                    try:
                        raw_response = client.search(query, params={"limit": limit})
                    except Exception:
                        raw_response = client.search(query)

            items = []
            if raw_response is not None:
                if hasattr(raw_response, "model_dump"):
                    dumped = raw_response.model_dump()
                    items = dumped.get("data", []) or dumped.get("results", []) or []
                elif hasattr(raw_response, "dict"):
                    dumped = raw_response.dict()
                    items = dumped.get("data", []) or dumped.get("results", []) or []
                elif isinstance(raw_response, dict):
                    items = raw_response.get("data", []) or raw_response.get("results", []) or []
                elif isinstance(raw_response, list):
                    items = raw_response

            if items:
                return self._normalize(items[:limit])
        except Exception as e:
            logger.warning(f"Firecrawl SDK search failed: {e}. Attempting REST fallback...")

        # Direct REST API fallback with retries
        url = "https://api.firecrawl.dev/v1/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"query": query, "limit": limit}

        for attempt in range(1, settings.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=settings.request_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", []) or data.get("results", []) or []
                    if isinstance(items, list) and items:
                        return self._normalize(items[:limit])
                elif resp.status_code == 429:
                    wait_time = 1.5 ** attempt
                    logger.warning(f"Firecrawl rate limit (429). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"Firecrawl API returned status {resp.status_code}")
                    break
            except Exception as ex:
                if attempt == settings.max_retries:
                    raise ex
                time.sleep(1.0)

        return []

    def _normalize(self, items: List[Any]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "dict"):
                item = item.dict()

            title = ""
            url = ""
            desc = ""
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or ""
                url = item.get("url") or item.get("link") or ""
                desc = item.get("description") or item.get("snippet") or item.get("markdown") or ""
                meta = item.get("metadata")
                if isinstance(meta, dict):
                    title = title or meta.get("title") or ""
                    desc = desc or meta.get("description") or ""

            if url and url != "None":
                results.append({
                    "title": title or "Untitled",
                    "url": url,
                    "description": " ".join(desc.split()) if desc else "No description available.",
                    "provider": "firecrawl",
                })
        return results


class DuckDuckGoSearchProvider(SearchProvider):
    """
    Zero-key, public DuckDuckGo search provider for guaranteed fallback availability.
    """
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Using DuckDuckGo fallback search for query: '{query}'")

        # Try duckduckgo_search package if installed
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=limit))
                if ddg_results:
                    return [
                        {
                            "title": r.get("title", "Untitled"),
                            "url": r.get("href", ""),
                            "description": r.get("body", ""),
                            "provider": "duckduckgo",
                        }
                        for r in ddg_results if r.get("href")
                    ]
        except Exception:
            pass

        # Standalone HTTP fallback against DuckDuckGo HTML endpoint
        endpoint = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        data = {"q": query}

        try:
            resp = requests.post(endpoint, headers=headers, data=data, timeout=settings.request_timeout)
            if resp.status_code != 200:
                logger.warning(f"DuckDuckGo search returned HTTP {resp.status_code}")
                return []

            html = resp.text
            results: List[Dict[str, Any]] = []

            # Extract result blocks using regex pattern
            result_pattern = re.compile(
                r'<a[^>]+class="[^"]*result__url[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>',
                re.IGNORECASE,
            )
            snippet_pattern = re.compile(
                r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>',
                re.IGNORECASE,
            )

            urls_found = result_pattern.findall(html)
            snippets_found = snippet_pattern.findall(html)

            for i, (raw_url, raw_title) in enumerate(urls_found[:limit]):
                clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                clean_snippet = ""
                if i < len(snippets_found):
                    clean_snippet = re.sub(r"<[^>]+>", "", snippets_found[i]).strip()

                # Extract real URL from DDG redirect wrapper if needed
                actual_url = raw_url.strip()
                if "uddg=" in actual_url:
                    m = re.search(r"uddg=([^&]+)", actual_url)
                    if m:
                        actual_url = unquote(m.group(1))

                if actual_url.startswith("http"):
                    results.append({
                        "title": clean_title or "Untitled Research Result",
                        "url": actual_url,
                        "description": clean_snippet or "No description available.",
                        "provider": "duckduckgo",
                    })

            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []


def search_with_fallback(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search using the configured primary search provider, with automated fallback
    to DuckDuckGo if the primary provider fails, rate limits, or is unconfigured.
    """
    clean_q = query.strip()
    if not clean_q:
        return []

    provider_name = (settings.search_provider or "firecrawl").lower()
    primary: SearchProvider
    fallback: SearchProvider = DuckDuckGoSearchProvider()

    if provider_name == "duckduckgo":
        return fallback.search(clean_q, limit=limit)

    primary = FirecrawlSearchProvider()

    try:
        results = primary.search(clean_q, limit=limit)
        if results:
            logger.info(f"Primary search (Firecrawl) returned {len(results)} results")
            return results
        logger.info("Primary search returned 0 results. Triggering DuckDuckGo fallback...")
    except Exception as ex:
        logger.warning(f"Primary search provider failed ({ex}). Switching to DuckDuckGo fallback...")

    # Fallback execution
    fallback_results = fallback.search(clean_q, limit=limit)
    logger.info(f"Fallback search (DuckDuckGo) returned {len(fallback_results)} results")
    return fallback_results
