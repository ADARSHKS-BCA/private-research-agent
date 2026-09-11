"""
Streaming adapter for the LangGraph Autonomous Research Agent.

Exposes the existing LangGraph execution as a Server-Sent Events (SSE) stream.
Safely extracts high-level progress status without exposing internal reasoning,
private chain-of-thought, or system prompts.
"""

import asyncio
import json
import logging
import queue
import re
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.agent.graph import create_research_graph
from app.agent.state import ResearchState, create_initial_state

logger = logging.getLogger(__name__)

# User-facing safe status labels mapping node names to clean display text
STEP_DISPLAY_CONFIG = {
    "plan_research": {
        "title": "Planning research",
        "description": "Formulating targeted search strategy",
    },
    "search_web": {
        "title": "Searching the web",
        "description": "Discovering authoritative web sources",
    },
    "scrape_sources": {
        "title": "Scraping relevant sources",
        "description": "Extracting clean full-text content",
    },
    "process_documents": {
        "title": "Processing documents",
        "description": "Structuring and chunking document text",
    },
    "index_documents": {
        "title": "Indexing knowledge base",
        "description": "Generating dense embeddings and storing vectors",
    },
    "retrieve_evidence": {
        "title": "Searching knowledge base",
        "description": "Retrieving most relevant evidence passages",
    },
    "evaluate_evidence": {
        "title": "Evaluating evidence",
        "description": "Checking evidence sufficiency for research question",
    },
    "generate_answer": {
        "title": "Generating answer",
        "description": "Synthesizing grounded answer with citations",
    },
    "validate_citations": {
        "title": "Validating citations",
        "description": "Verifying citations match legitimate source URLs",
    },
}

NEXT_STEP_MAP = {
    "plan_research": "search_web",
    "search_web": "scrape_sources",
    "scrape_sources": "process_documents",
    "process_documents": "index_documents",
    "index_documents": "retrieve_evidence",
    "retrieve_evidence": "evaluate_evidence",
}


def _format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format SSE event string according to standard."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _split_into_streaming_tokens(text: str) -> List[str]:
    """
    Split text into words and tokens preserving whitespace and formatting
    for natural progressive typing.
    """
    if not text:
        return []
    # Match words, spaces, newlines, and punctuation tokens
    tokens = re.findall(r"\S+|\s+", text)
    return tokens


async def stream_research_events(
    question: str,
    max_iterations: int = 3,
) -> AsyncGenerator[str, None]:
    """
    Stream safe progress events, answer tokens, and validated sources
    from the existing LangGraph execution over SSE.
    """
    clean_q = question.strip() if question else ""
    if not clean_q:
        yield _format_sse_event("error", {"message": "Question cannot be empty."})
        yield _format_sse_event("done", {})
        return

    start_time = time.time()

    # Initial status
    yield _format_sse_event(
        "status",
        {
            "step": "init",
            "title": "Understanding research question",
            "state": "completed",
            "details": "Initialized autonomous research workflow",
        },
    )

    # Indicate first step is running
    first_step_conf = STEP_DISPLAY_CONFIG["plan_research"]
    yield _format_sse_event(
        "status",
        {
            "step": "plan_research",
            "title": first_step_conf["title"],
            "state": "running",
            "details": "Analyzing query and formulating search strategy",
        },
    )

    event_queue: queue.Queue = queue.Queue()
    _SENTINEL = object()

    def run_graph_sync():
        try:
            graph = create_research_graph()
            initial_state = create_initial_state(question=clean_q, max_iterations=max_iterations)

            # Stream node updates from LangGraph
            for event in graph.stream(initial_state, stream_mode="updates"):
                event_queue.put(("node_update", event))

            event_queue.put((_SENTINEL, None))
        except Exception as ex:
            logger.exception("Error during LangGraph execution in worker thread")
            event_queue.put(("error", ex))
            event_queue.put((_SENTINEL, None))

    # Execute LangGraph in background worker thread to prevent blocking asyncio loop
    worker_thread = threading.Thread(target=run_graph_sync, daemon=True)
    worker_thread.start()

    answer_text = ""
    validated_sources: List[Dict[str, Any]] = []

    try:
        while True:
            # Check for events from worker thread without busy-waiting
            try:
                item_type, item_data = await asyncio.to_thread(event_queue.get, timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue

            if item_type is _SENTINEL:
                break

            if item_type == "error":
                logger.error(f"LangGraph execution encountered error: {item_data}")
                yield _format_sse_event(
                    "error",
                    {"message": "Something went wrong while researching this question. Please try again."},
                )
                break

            if item_type == "node_update":
                # item_data is a dict of {node_name: state_update}
                for node_name, state_update in item_data.items():
                    config = STEP_DISPLAY_CONFIG.get(node_name, {
                        "title": node_name.replace("_", " ").title(),
                        "description": "",
                    })

                    details = ""
                    if node_name == "plan_research":
                        queries = state_update.get("search_queries", [])
                        details = f"Generated {len(queries)} search query" if len(queries) == 1 else f"Generated {len(queries)} search queries"
                    elif node_name == "search_web":
                        results = state_update.get("search_results", [])
                        details = f"Discovered {len(results)} relevant web sources"
                    elif node_name == "scrape_sources":
                        scraped = state_update.get("scraped_documents", [])
                        successful = [d for d in scraped if getattr(d, "success", False)]
                        details = f"Extracted content from {len(successful)} pages"
                    elif node_name == "process_documents":
                        chunks = state_update.get("chunks", [])
                        details = f"Created {len(chunks)} structured chunks"
                    elif node_name == "index_documents":
                        details = "Vectors stored in Qdrant collection"
                    elif node_name == "retrieve_evidence":
                        retrieved = state_update.get("retrieved_documents", [])
                        details = f"Retrieved top {len(retrieved)} evidence passages"
                    elif node_name == "evaluate_evidence":
                        is_sufficient = state_update.get("evidence_sufficient", False)
                        iteration = state_update.get("research_iteration", 1)
                        if is_sufficient:
                            details = f"Evidence verified (Iteration {iteration})"
                        else:
                            details = f"Additional context required (Entering loop {iteration})"
                    elif node_name == "generate_answer":
                        answer_text = state_update.get("answer", "")
                        details = "Answer synthesis completed"
                    elif node_name == "validate_citations":
                        sources_list = state_update.get("sources", [])
                        validated_sources = sources_list
                        details = f"Verified {len(sources_list)} source citations"

                    # Emit completion status for current node
                    yield _format_sse_event(
                        "status",
                        {
                            "step": node_name,
                            "title": config["title"],
                            "state": "completed",
                            "details": details,
                        },
                    )

                    # Emit 'running' status for expected next node
                    next_node = NEXT_STEP_MAP.get(node_name)
                    if next_node and next_node in STEP_DISPLAY_CONFIG:
                        next_conf = STEP_DISPLAY_CONFIG[next_node]
                        yield _format_sse_event(
                            "status",
                            {
                                "step": next_node,
                                "title": next_conf["title"],
                                "state": "running",
                                "details": next_conf["description"],
                            },
                        )
                    elif node_name == "evaluate_evidence":
                        is_sufficient = state_update.get("evidence_sufficient", False)
                        target_next = "generate_answer" if is_sufficient else "plan_research"
                        next_conf = STEP_DISPLAY_CONFIG[target_next]
                        yield _format_sse_event(
                            "status",
                            {
                                "step": target_next,
                                "title": next_conf["title"],
                                "state": "running",
                                "details": next_conf["description"],
                            },
                        )
                    elif node_name == "generate_answer":
                        next_conf = STEP_DISPLAY_CONFIG["validate_citations"]
                        yield _format_sse_event(
                            "status",
                            {
                                "step": "validate_citations",
                                "title": next_conf["title"],
                                "state": "running",
                                "details": next_conf["description"],
                            },
                        )

                    # When answer is generated, stream tokens progressively
                    if node_name == "generate_answer" and answer_text:
                        tokens = _split_into_streaming_tokens(answer_text)
                        for token in tokens:
                            yield _format_sse_event("token", {"text": token})
                            # Natural progressive typing delay
                            await asyncio.sleep(0.012)

                    # When citations are validated, emit sources event
                    if node_name == "validate_citations":
                        yield _format_sse_event(
                            "sources",
                            {"sources": validated_sources},
                        )

        elapsed = round(time.time() - start_time, 2)
        yield _format_sse_event("done", {"elapsed_seconds": elapsed})

    except Exception as e:
        logger.exception(f"Unhandled error in research stream: {e}")
        yield _format_sse_event(
            "error",
            {"message": "Something went wrong while researching this question. Please try again."},
        )
        yield _format_sse_event("done", {})
