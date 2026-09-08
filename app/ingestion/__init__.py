from app.ingestion.chunker import DocumentChunk, StructureAwareChunker, chunk_document
from app.ingestion.cleaner import CleanedDocument, Cleaner, clean_document
from app.ingestion.embedder import LocalEmbedder, embed_chunks, get_embedder
from app.ingestion.qdrant_store import QdrantStore, store_chunks
from app.ingestion.scraper import ScrapedDocument, Scraper, generate_document_id, scrape_url


def ingest_url(url: str, collection_name: str = None, verbose: bool = False):
    """Lazy import to avoid circular runpy warnings when executing pipeline directly."""
    from app.ingestion.pipeline import ingest_url as _ingest_url
    return _ingest_url(url=url, collection_name=collection_name, verbose=verbose)


__all__ = [
    "ScrapedDocument",
    "Scraper",
    "generate_document_id",
    "scrape_url",
    "CleanedDocument",
    "Cleaner",
    "clean_document",
    "DocumentChunk",
    "StructureAwareChunker",
    "chunk_document",
    "LocalEmbedder",
    "get_embedder",
    "embed_chunks",
    "QdrantStore",
    "store_chunks",
    "ingest_url",
]
