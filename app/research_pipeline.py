import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import get_embedder
from app.ingestion.qdrant_store import QdrantStore
from app.processing.cleaner import process_scraped_documents
from app.rag.answer import generate_answer_from_retrieved_chunks
from app.retrieval.search import search
from app.web.scraper import scrape_search_results
from app.web.search import search_web

logger = logging.getLogger(__name__)


def run_research(
    question: str,
    num_urls: int = 5,
    top_k_retrieval: int = 5,
    collection_name: Optional[str] = None,
    stream: bool = True,
) -> Dict[str, Any]:
    """
    Execute the Complete Deterministic Dynamic Research Pipeline:
    1. Search Web (Firecrawl)
    2. Scrape multiple discovered URLs (Firecrawl)
    3. Clean content (Rule-based)
    4. Generate metadata, deterministic doc_ids & content hashes
    5. Structure-aware chunking (approx 800 tokens, 100 overlap)
    6. Generate dense vector embeddings locally (BAAI/bge-m3)
    7. Idempotent storage in Qdrant (research_documents)
    8. Dense vector retrieval for the question
    9. Grounded answer generation with Groq (openai/gpt-oss-20b)
    10. Citation validation ([S1]/[S2] mapped to trusted URLs)
    """
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Research question cannot be empty.")

    target_collection = collection_name or settings.qdrant_collection
    start_time = time.time()

    print("\n" + "=" * 70)
    print("  PRIVATE RESEARCH AGENT - DYNAMIC RESEARCH PIPELINE")
    print("=" * 70)
    print(f"Research Question: \"{clean_question}\"\n")

    # ----------------------------------------------------
    # STEP 1: Search Web
    # ----------------------------------------------------
    print(f"[1/10] Searching web for: '{clean_question}' (limit: {num_urls})...")
    search_results = search_web(query=clean_question, limit=num_urls)

    # ----------------------------------------------------
    # STEP 2: Found URLs
    # ----------------------------------------------------
    if not search_results:
        print("[2/10] No URLs found from web search.")
        return {
            "question": clean_question,
            "urls_discovered": [],
            "documents_scraped": 0,
            "chunks_stored": 0,
            "answer": "No web results were found for this query.",
            "sources": [],
        }

    print(f"[2/10] Discovered {len(search_results)} relevant URLs:")
    for idx, item in enumerate(search_results, start=1):
        print(f"       [{idx}] {item.get('title', 'Untitled')} ({item.get('url')})")
    print()

    # ----------------------------------------------------
    # STEP 3: Scrape URLs
    # ----------------------------------------------------
    print(f"[3/10] Scraping {len(search_results)} URLs via Firecrawl...")
    scraped_docs = scrape_search_results(search_results)
    successful_scrapes = [d for d in scraped_docs if d.success]
    print(f"       Scraped {len(successful_scrapes)}/{len(scraped_docs)} pages successfully.\n")

    if not successful_scrapes:
        print("[Error] Failed to scrape any content from discovered URLs.")
        return {
            "question": clean_question,
            "urls_discovered": [s.get("url") for s in search_results],
            "documents_scraped": 0,
            "chunks_stored": 0,
            "answer": "Failed to scrape content from discovered web pages.",
            "sources": [],
        }

    # ----------------------------------------------------
    # STEP 4 & 5: Clean Documents, Metadata & Content Hashes
    # ----------------------------------------------------
    print("[4/10] Cleaning documents & removing web boilerplate...")
    print("[5/10] Creating deterministic document IDs & content hashes...")
    processed_docs = process_scraped_documents(successful_scrapes)
    print(f"       Processed {len(processed_docs)} unique clean documents.\n")

    if not processed_docs:
        print("[Error] No documents remained after cleaning and duplicate filtering.")
        return {
            "question": clean_question,
            "documents_scraped": len(successful_scrapes),
            "chunks_stored": 0,
            "answer": "No substantive content remained after document cleaning.",
            "sources": [],
        }

    # ----------------------------------------------------
    # STEP 6: Chunking
    # ----------------------------------------------------
    print(f"[6/10] Chunking documents (size: {settings.chunk_size} tokens, overlap: {settings.chunk_overlap})...")
    chunks = chunk_documents(processed_docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    print(f"       Generated {len(chunks)} structure-aware chunks.\n")

    if not chunks:
        print("[Error] No chunks were created from the processed documents.")
        return {
            "question": clean_question,
            "documents_scraped": len(processed_docs),
            "chunks_stored": 0,
            "answer": "No content chunks could be extracted from the documents.",
            "sources": [],
        }

    # ----------------------------------------------------
    # STEP 7: Embeddings (Local BGE-M3)
    # ----------------------------------------------------
    print(f"[7/10] Generating local dense embeddings with {settings.embedding_model}...")
    embedder = get_embedder(model_name=settings.embedding_model)
    embeddings = embedder.embed_chunks(chunks)
    print(f"       Generated {len(embeddings)} vectors (dimension: {embedder.get_embedding_dimension()}).\n")

    # ----------------------------------------------------
    # STEP 8: Store in Qdrant
    # ----------------------------------------------------
    print(f"[8/10] Storing chunks & metadata in Qdrant ('{target_collection}')...")
    store = QdrantStore(collection_name=target_collection)
    chunks_stored = store.upsert_chunks(chunks=chunks, embeddings=embeddings)
    print(f"       Indexed {chunks_stored} points in Qdrant collection '{target_collection}'.\n")

    # ----------------------------------------------------
    # STEP 9: Retrieve Evidence
    # ----------------------------------------------------
    print(f"[9/10] Retrieving top {top_k_retrieval} relevant evidence chunks for question...")
    retrieved_results = search(clean_question, top_k=top_k_retrieval)
    print(f"       Retrieved {len(retrieved_results)} chunks from Qdrant.\n")

    # ----------------------------------------------------
    # STEP 10: Generate Grounded Answer & Validate Citations
    # ----------------------------------------------------
    provider = settings.llm_provider.upper()
    model_name = settings.groq_model if settings.llm_provider == "groq" else settings.ollama_model
    print(f"[10/10] Generating grounded answer with {provider} ({model_name}) & validating citations...")

    rag_output = generate_answer_from_retrieved_chunks(
        query=clean_question,
        results=retrieved_results,
        stream=stream,
    )

    elapsed = round(time.time() - start_time, 2)
    print(f"Pipeline completed in {elapsed}s.")

    return {
        "question": clean_question,
        "urls_discovered": [s.get("url") for s in search_results],
        "documents_scraped": len(processed_docs),
        "chunks_stored": chunks_stored,
        "retrieved_chunks": len(retrieved_results),
        "answer": rag_output.get("answer", ""),
        "sources": rag_output.get("sources", []),
        "invalid_citations": rag_output.get("invalid_citations", []),
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Private Research Agent - Complete Deterministic Pipeline")
    parser.add_argument("question", nargs="?", type=str, help="Research question to investigate", default=None)
    parser.add_argument("--urls", type=int, help="Number of URLs to search and scrape", default=4)
    parser.add_argument("--top_k", type=int, help="Number of chunks to retrieve for synthesis", default=4)
    args = parser.parse_args()

    question = args.question
    if not question:
        print("=" * 70)
        print("  PRIVATE RESEARCH AGENT")
        print("  (Type 'quit', 'exit', 'bye', or 'stop' to end)")
        print("=" * 70)

        while True:
            try:
                user_q = input("\nWhat would you like to research? ").strip()
                if not user_q:
                    continue
                if user_q.lower() in {"quit", "exit", "bye", "stop", "q"}:
                    print("\nEnding research session. Goodbye!\n")
                    break

                run_research(
                    question=user_q,
                    num_urls=args.urls,
                    top_k_retrieval=args.top_k,
                    stream=True,
                )
            except (KeyboardInterrupt, EOFError):
                print("\n\nSession interrupted. Goodbye!\n")
                break
    else:
        run_research(
            question=question,
            num_urls=args.urls,
            top_k_retrieval=args.top_k,
            stream=True,
        )


if __name__ == "__main__":
    main()
