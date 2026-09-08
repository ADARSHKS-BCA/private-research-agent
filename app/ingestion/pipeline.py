import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Ensure project root is in sys.path when running as a script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.ingestion.chunker import chunk_document
from app.ingestion.cleaner import clean_document
from app.ingestion.embedder import get_embedder
from app.ingestion.qdrant_store import QdrantStore
from app.ingestion.scraper import scrape_url

logger = logging.getLogger(__name__)


def ingest_url(
    url: str,
    collection_name: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute the complete 5-stage document ingestion pipeline:
    1. Scrape (Firecrawl) -> Raw Markdown -> save to data/raw/<document_id>.md
    2. Clean -> Cleaned Document -> save to data/processed/<document_id>.md
    3. Chunk -> Structure-aware chunks with metadata
    4. Embed -> Generate local vector embeddings (BAAI/bge-m3)
    5. Store -> Upsert vectors, payloads & text into Qdrant
    """
    target_collection = collection_name or settings.qdrant_collection

    if verbose:
        print(f"[1/5] Scraping...")
    logger.info(f"Stage 1/5: Scraping {url}")
    scraped_doc = scrape_url(url)

    if verbose:
        print(f"[2/5] Cleaning...")
    logger.info(f"Stage 2/5: Cleaning {scraped_doc.document_id}")
    cleaned_doc = clean_document(scraped_doc)

    if verbose:
        print(f"[3/5] Chunking...")
    logger.info(f"Stage 3/5: Chunking {cleaned_doc.document_id}")
    chunks = chunk_document(cleaned_doc)

    if not chunks:
        raise ValueError(f"No chunks were generated for document {scraped_doc.document_id}")

    if verbose:
        print(f"[4/5] Embedding...")
    logger.info(f"Stage 4/5: Generating embeddings for {len(chunks)} chunks")
    embedder = get_embedder()
    embeddings = embedder.embed_chunks(chunks)

    if verbose:
        print(f"[5/5] Storing in Qdrant...")
    logger.info(f"Stage 5/5: Storing {len(chunks)} chunks in Qdrant collection '{target_collection}'")
    store = QdrantStore(collection_name=target_collection)
    chunks_stored = store.upsert_chunks(chunks=chunks, embeddings=embeddings)

    result = {
        "document_id": scraped_doc.document_id,
        "url": url,
        "title": scraped_doc.title,
        "chunks_created": chunks_stored,
        "collection": target_collection,
        "raw_path": scraped_doc.raw_file_path,
        "processed_path": cleaned_doc.processed_file_path,
    }

    return result


def main():
    """CLI entrypoint for running ingestion pipeline directly."""
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingestion.pipeline <URL>")
        print("Example: python -m app.ingestion.pipeline https://arxiv.org/abs/2005.11401")
        sys.exit(1)

    target_url = sys.argv[1].strip()

    try:
        result = ingest_url(target_url, verbose=True)

        print()
        print(f"Document ID: {result['document_id']}")
        print(f"Chunks created: {result['chunks_created']}")
        print(f"Qdrant collection: {result['collection']}")
        print()
        print("Ingestion completed successfully.")
    except Exception as e:
        print(f"\nIngestion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
