import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv()


def search_with_rest_api(query: str, api_key: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Direct REST API call to Firecrawl /v1/search endpoint.
    Guarantees standard JSON parsing across any SDK version.
    """
    url = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "limit": limit,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            return []
        data = resp.json()
        items = data.get("data", []) or data.get("results", []) or []
        if isinstance(items, list):
            return items
        return []
    except Exception:
        return []


def get_firecrawl_client(api_key: Optional[str] = None):
    key = api_key or os.getenv("FIRECRAWL_API_KEY")
    try:
        from firecrawl import FirecrawlApp
        return FirecrawlApp(api_key=key)
    except (ImportError, AttributeError):
        from firecrawl import Firecrawl
        return Firecrawl(api_key=key)


def search_web(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using Firecrawl Search API.

    Args:
        query: Natural-language search question or keyword string.
        limit: Maximum number of search results to return (default: 5).

    Returns:
        List of result dictionaries containing 'title', 'url', and 'description'.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY is not set. Please add it to your .env file.")

    items: List[Any] = []

    # Strategy 1: Try Firecrawl Python SDK
    try:
        client = get_firecrawl_client(api_key=api_key)

        if hasattr(client, "search"):
            try:
                raw_response = client.search(clean_query, limit=limit)
            except TypeError:
                try:
                    raw_response = client.search(clean_query, params={"limit": limit})
                except Exception:
                    raw_response = client.search(clean_query)
            except Exception:
                raw_response = None

            if raw_response is not None:
                # Handle Pydantic model dump
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
                else:
                    items = getattr(raw_response, "data", None) or getattr(raw_response, "results", None) or []
    except Exception:
        items = []

    # Strategy 2: Direct REST API Fallback
    if not items:
        items = search_with_rest_api(clean_query, api_key=api_key, limit=limit)

    normalized_results: List[Dict[str, Any]] = []

    for item in items[:limit]:
        title = ""
        url = ""
        description = ""

        # Convert Pydantic item to dict if applicable
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        elif hasattr(item, "dict"):
            item = item.dict()

        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "")
            url = str(item.get("url") or item.get("link") or "")
            description = str(
                item.get("description")
                or item.get("snippet")
                or item.get("markdown")
                or item.get("summary")
                or ""
            )
            meta = item.get("metadata")
            if isinstance(meta, dict):
                if not title:
                    title = str(meta.get("title") or "")
                if not description:
                    description = str(meta.get("description") or "")
        else:
            title = str(getattr(item, "title", None) or getattr(item, "name", None) or "")
            url = str(getattr(item, "url", None) or getattr(item, "link", None) or "")
            description = str(
                getattr(item, "description", None)
                or getattr(item, "snippet", None)
                or getattr(item, "markdown", None)
                or getattr(item, "summary", None)
                or ""
            )
            meta = getattr(item, "metadata", None)
            if meta:
                if isinstance(meta, dict):
                    if not title:
                        title = str(meta.get("title") or "")
                    if not description:
                        description = str(meta.get("description") or "")
                else:
                    if not title:
                        title = str(getattr(meta, "title", None) or "")
                    if not description:
                        description = str(getattr(meta, "description", None) or "")

        # Clean description whitespace
        if description and description != "None":
            description = " ".join(description.split())
        else:
            description = "No description available."

        if url and url != "None":
            normalized_results.append({
                "title": title if (title and title != "None") else "Untitled",
                "url": url,
                "description": description,
            })

    return normalized_results


def main():
    parser = argparse.ArgumentParser(description="Firecrawl Web Search Module")
    parser.add_argument("--query", type=str, help="Search query string", default=None)
    parser.add_argument("--limit", type=int, help="Maximum number of results", default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("  Private Research Agent - Web Search (Firecrawl)")
    print("=" * 70)

    query = args.query
    if not query:
        query = input("\nWhat do you want to research? ").strip()

    if not query:
        print("Empty research query. Exiting.")
        return

    print(f"\nSearching the web for: '{query}' (limit: {args.limit})...\n")

    try:
        results = search_web(query=query, limit=args.limit)

        if not results:
            print("No results found for this query.")
            return

        for i, result in enumerate(results, start=1):
            print("=" * 70)
            print(f"RESULT {i}")
            print("=" * 70)
            print(f"Title:       {result.get('title')}")
            print(f"URL:         {result.get('url')}")
            print(f"Description: {result.get('description')}")
            print()

    except Exception as e:
        print(f"[Error] Web search failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()