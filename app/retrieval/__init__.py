def search(query: str, top_k: int = 5):
    """Lazy import to avoid runpy warnings when executing search directly."""
    from app.retrieval.search import search as _search
    return _search(query=query, top_k=top_k)

__all__ = ["search"]
