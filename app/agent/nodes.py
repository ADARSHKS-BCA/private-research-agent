"""
LangGraph Nodes for the Autonomous Private Research Agent.

Reuses existing pipeline modules:
- Firecrawl Search (app.web.search)
- Scraper (app.web.scraper)
- Rule-based Cleaner & Metadata (app.processing.cleaner)
- Structure-Aware Chunker (app.ingestion.chunker)
- Local BGE-M3 Embeddings (app.ingestion.embedder)
- Qdrant Vector Store (app.ingestion.qdrant_store)
- Dense Retrieval (app.retrieval.search)
- Grounded Groq Generation (app.rag.answer)
- Citation Validation (app.rag.citations)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    format_answer_prompt,
    format_evaluator_prompt,
    format_planner_prompt,
)
from app.agent.state import ResearchState
from app.config import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import get_embedder
from app.ingestion.qdrant_store import QdrantStore
from app.processing.cleaner import process_scraped_documents
from app.rag.answer import get_groq_available_models
from app.rag.citations import (
    extract_citations,
    format_prompt_context_with_sources,
    validate_citations as validate_citations_func,
)
from app.retrieval.search import search
from app.web.scraper import scrape_search_results
from app.web.search import search_web

logger = logging.getLogger(__name__)


def _call_llm(
    prompt: str,
    system_prompt: str = "You are a helpful research assistant.",
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """
    Invoke Groq (or local Ollama fallback) using existing project configuration.
    """
    api_key = settings.groq_api_key
    provider = (settings.llm_provider or "groq").lower()

    if provider == "groq" and api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            target_model = settings.groq_model or "openai/gpt-oss-20b"
            
            # Verify model availability
            available_models = get_groq_available_models(client)
            candidate_models: List[str] = []

            if target_model in available_models:
                candidate_models.append(target_model)
            else:
                matched = [m for m in available_models if target_model.lower() in m.lower()]
                if matched:
                    candidate_models.extend(matched)
                else:
                    candidate_models.append(target_model)

            standard_fallbacks = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ]
            for fb in standard_fallbacks:
                if fb not in candidate_models and (not available_models or fb in available_models):
                    candidate_models.append(fb)

            last_err = None
            for model_id in candidate_models:
                try:
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    return completion.choices[0].message.content.strip()
                except Exception as ex:
                    last_err = ex
                    continue
            if last_err:
                logger.warning(f"Groq invocation failed across candidates: {last_err}")
        except Exception as e:
            logger.warning(f"Groq client failed: {e}. Attempting Ollama fallback...")

    # Fallback to Ollama if configured or Groq fails
    try:
        import ollama
        host = settings.ollama_base_url or "http://localhost:11434"
        client = ollama.Client(host=host)
        chosen_model = settings.ollama_model or "qwen3:4b"
        resp = client.chat(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM call failed on both Groq and Ollama: {e}")
        return ""


def _extract_json(text: str) -> Any:
    """Extract and parse JSON object or array from LLM output."""
    if not text:
        return None

    cleaned = text.strip()
    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try finding JSON array or object
    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass

    return None


# =====================================================================
# NODE 1: plan_research
# =====================================================================
def plan_research(state: ResearchState) -> Dict[str, Any]:
    """
    Analyze the question and generate 1-3 targeted search queries.
    If subsequent iteration, generate follow-up queries based on missing evidence.
    """
    question = state.get("question", "")
    iteration = state.get("research_iteration", 1)
    previous_queries = state.get("search_queries", [])
    retrieved_docs = state.get("retrieved_documents", [])

    # Format brief evidence summary for follow-up iterations
    evidence_summary = ""
    if retrieved_docs:
        snippets = []
        for d in retrieved_docs[:3]:
            payload = getattr(d, "payload", {}) or (d if isinstance(d, dict) else {})
            text = payload.get("text", "")[:150]
            if text:
                snippets.append(f"- {text}...")
        evidence_summary = "\n".join(snippets)

    prompt = format_planner_prompt(
        question=question,
        iteration=iteration,
        previous_queries=previous_queries,
        evidence_summary=evidence_summary,
    )

    llm_resp = _call_llm(
        prompt=prompt,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=250,
    )

    parsed = _extract_json(llm_resp)
    new_queries: List[str] = []

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and item.strip():
                clean_item = item.strip()
                if clean_item not in previous_queries and clean_item not in new_queries:
                    new_queries.append(clean_item)
    elif isinstance(parsed, dict) and "queries" in parsed and isinstance(parsed["queries"], list):
        for item in parsed["queries"]:
            if isinstance(item, str) and item.strip():
                clean_item = item.strip()
                if clean_item not in previous_queries and clean_item not in new_queries:
                    new_queries.append(clean_item)

    # Fallback if no valid queries could be parsed
    if not new_queries:
        if iteration == 1 or not previous_queries:
            new_queries = [question]
        else:
            new_queries = [f"{question} detailed overview"]

    # Limit to maximum 3 queries
    new_queries = new_queries[:3]

    print(f"[PLAN] Generated {len(new_queries)} research queries: {new_queries}", flush=True)

    return {
        "search_queries": previous_queries + new_queries,
    }


# =====================================================================
# NODE 2: search_web
# =====================================================================
def search_web_node(state: ResearchState) -> Dict[str, Any]:
    """
    Execute web search using the existing Firecrawl implementation.
    Deduplicates URLs across iterations.
    """
    search_queries = state.get("search_queries", [])
    existing_results = state.get("search_results", [])
    errors = list(state.get("errors", []))

    seen_urls = {r.get("url") for r in existing_results if isinstance(r, dict) and r.get("url")}
    new_results: List[Dict[str, Any]] = []

    # Search each recent query (limit 3-4 results per query)
    # Focus on queries generated in current iteration (up to last 3)
    queries_to_run = search_queries[-3:] if search_queries else [state.get("question", "")]

    for query_str in queries_to_run:
        try:
            items = search_web(query=query_str, limit=4)
            for item in items:
                url = item.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    new_results.append(item)
        except Exception as e:
            err_msg = f"Search error for query '{query_str}': {e}"
            logger.warning(err_msg)
            errors.append(err_msg)

    total_results = existing_results + new_results
    print(f"[SEARCH] Found {len(new_results)} new URLs (Total discovered: {len(total_results)})", flush=True)

    return {
        "search_results": total_results,
        "errors": errors,
    }


# =====================================================================
# NODE 3: scrape_sources
# =====================================================================
def scrape_sources(state: ResearchState) -> Dict[str, Any]:
    """
    Scrape discovered URLs using the existing Firecrawl scraper.
    Gracefully handles failures per URL.
    """
    search_results = state.get("search_results", [])
    existing_scraped = state.get("scraped_documents", [])
    errors = list(state.get("errors", []))

    scraped_urls = {
        getattr(d, "url", None) or (d.get("url") if isinstance(d, dict) else None)
        for d in existing_scraped
    }

    # Filter unscraped search results, limiting to top 4 per iteration to stay efficient
    unscraped_items = [
        item for item in search_results
        if item.get("url") and item.get("url") not in scraped_urls
    ][:4]

    if not unscraped_items:
        print("[SCRAPE] No new URLs to scrape.", flush=True)
        return {
            "scraped_documents": existing_scraped,
            "errors": errors,
        }

    try:
        newly_scraped = scrape_search_results(unscraped_items)
    except Exception as e:
        err_msg = f"Scraping batch failed: {e}"
        logger.warning(err_msg)
        errors.append(err_msg)
        newly_scraped = []

    successful = [d for d in newly_scraped if getattr(d, "success", False)]
    print(f"[SCRAPE] Scraped {len(successful)} sources successfully ({len(newly_scraped)} attempted)", flush=True)

    all_scraped = existing_scraped + newly_scraped
    return {
        "scraped_documents": all_scraped,
        "errors": errors,
    }


# =====================================================================
# NODE 4: process_documents
# =====================================================================
def process_documents(state: ResearchState) -> Dict[str, Any]:
    """
    Clean scraped documents, attach metadata, and chunk using existing modules.
    """
    scraped_docs = state.get("scraped_documents", [])
    errors = list(state.get("errors", []))

    successful_scrapes = [d for d in scraped_docs if getattr(d, "success", False)]

    try:
        processed_docs = process_scraped_documents(successful_scrapes)
        chunks = chunk_documents(
            processed_docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except Exception as e:
        err_msg = f"Document processing/chunking failed: {e}"
        logger.warning(err_msg)
        errors.append(err_msg)
        processed_docs = []
        chunks = []

    print(f"[PROCESS] Created {len(chunks)} chunks from {len(processed_docs)} cleaned documents", flush=True)

    return {
        "cleaned_documents": processed_docs,
        "chunks": chunks,
        "errors": errors,
    }


# =====================================================================
# NODE 5: index_documents
# =====================================================================
def index_documents(state: ResearchState) -> Dict[str, Any]:
    """
    Generate local dense embeddings (BGE-M3) and store in Qdrant.
    """
    chunks = state.get("chunks", [])
    errors = list(state.get("errors", []))

    if not chunks:
        print("[INDEX] No chunks to index.", flush=True)
        return {"errors": errors}

    try:
        embedder = get_embedder(model_name=settings.embedding_model)
        embeddings = embedder.embed_chunks(chunks)
        store = QdrantStore(collection_name=settings.qdrant_collection)
        stored_count = store.upsert_chunks(chunks=chunks, embeddings=embeddings)
        print(f"[INDEX] Stored {stored_count} chunks in Qdrant", flush=True)
    except Exception as e:
        err_msg = f"Indexing into Qdrant failed: {e}"
        logger.warning(err_msg)
        errors.append(err_msg)

    return {
        "errors": errors,
    }


# =====================================================================
# NODE 6: retrieve_evidence
# =====================================================================
def retrieve_evidence(state: ResearchState) -> Dict[str, Any]:
    """
    Retrieve the most relevant chunks from Qdrant for the original question.
    """
    question = state.get("question", "")
    errors = list(state.get("errors", []))

    retrieved: List[Any] = []
    try:
        retrieved = search(question, top_k=6)
    except Exception as e:
        err_msg = f"Qdrant retrieval search error: {e}"
        logger.warning(err_msg)
        errors.append(err_msg)

    print(f"[RETRIEVE] Retrieved {len(retrieved)} chunks from Qdrant", flush=True)

    return {
        "retrieved_documents": retrieved,
        "errors": errors,
    }


# =====================================================================
# NODE 7: evaluate_evidence
# =====================================================================
def evaluate_evidence(state: ResearchState) -> Dict[str, Any]:
    """
    Assess whether retrieved evidence is sufficient to answer the research question.
    Enforces maximum research iteration bounds to guarantee termination.
    """
    question = state.get("question", "")
    retrieved = state.get("retrieved_documents", [])
    iteration = state.get("research_iteration", 1)
    max_iterations = state.get("max_iterations", 3)

    # Edge case 1: Hard iteration cap reached -> proceed to generate answer
    if iteration >= max_iterations:
        print(f"[EVALUATE] Max iterations ({max_iterations}) reached. Proceeding to answer generation.", flush=True)
        return {
            "evidence_sufficient": True,
            "research_iteration": iteration,
        }

    # Edge case 2: No chunks retrieved at all -> insufficient
    if not retrieved:
        print(f"[EVALUATE] Evidence sufficient: False (No chunks retrieved. Iteration {iteration}/{max_iterations})", flush=True)
        return {
            "evidence_sufficient": False,
            "research_iteration": iteration + 1,
        }

    # Prepare evidence context for evaluation
    citations = extract_citations(retrieved)
    context = format_prompt_context_with_sources(citations, max_text_len=600)

    prompt = format_evaluator_prompt(question=question, context=context)

    llm_resp = _call_llm(
        prompt=prompt,
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=200,
    )

    parsed = _extract_json(llm_resp)
    is_sufficient = False

    if isinstance(parsed, dict) and "sufficient" in parsed:
        is_sufficient = bool(parsed["sufficient"])
    else:
        # Heuristic fallback if LLM returned non-JSON
        lower = llm_resp.lower()
        if "sufficient\": true" in lower or "true" in lower:
            is_sufficient = True
        elif len(retrieved) >= 3 and len(context) > 500:
            is_sufficient = True

    print(f"[EVALUATE] Evidence sufficient: {is_sufficient} (Iteration {iteration}/{max_iterations})", flush=True)

    next_iteration = iteration if is_sufficient else iteration + 1

    return {
        "evidence_sufficient": is_sufficient,
        "research_iteration": next_iteration,
    }


# =====================================================================
# NODE 8: generate_answer
# =====================================================================
def generate_answer(state: ResearchState) -> Dict[str, Any]:
    """
    Synthesize a factually grounded answer citing [S1], [S2] source markers.
    Uses Groq with openai/gpt-oss-20b.
    """
    question = state.get("question", "")
    retrieved = state.get("retrieved_documents", [])
    errors = list(state.get("errors", []))

    print(f"[GENERATE] Generating answer with {settings.llm_provider.upper()} ({settings.groq_model if settings.llm_provider == 'groq' else settings.ollama_model})", flush=True)

    if not retrieved:
        no_info = "I don't have enough information in the provided sources to answer this question."
        return {
            "answer": no_info,
            "errors": errors,
        }

    citations = extract_citations(retrieved)
    context = format_prompt_context_with_sources(citations, max_text_len=1200)

    prompt = format_answer_prompt(question=question, context=context)

    answer_text = _call_llm(
        prompt=prompt,
        system_prompt=ANSWER_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=800,
    )

    if not answer_text.strip():
        answer_text = "I don't have enough information in the provided sources to answer this question."

    return {
        "answer": answer_text,
        "errors": errors,
    }


# =====================================================================
# NODE 9: validate_citations
# =====================================================================
def validate_citations(state: ResearchState) -> Dict[str, Any]:
    """
    Validate cited source IDs ([S1], [S2]) against actual retrieved Qdrant chunks.
    Rejects fabricated sources and maps IDs to legitimate URLs.
    """
    answer_text = state.get("answer", "")
    retrieved = state.get("retrieved_documents", [])
    errors = list(state.get("errors", []))

    citations = extract_citations(retrieved)
    valid_citations, invalid_ids = validate_citations_func(answer_text, citations)

    if invalid_ids:
        err_msg = f"Filtered out invalid citation IDs: {', '.join(invalid_ids)}"
        logger.info(err_msg)
        errors.append(err_msg)

    print(f"[CITATIONS] Validated {len(valid_citations)} sources", flush=True)

    return {
        "sources": [c.to_dict() for c in valid_citations],
        "errors": errors,
    }
