"""
Streaming adapter for the LangGraph Autonomous Research Agent.

Exposes the existing LangGraph execution as a Server-Sent Events (SSE) stream.
Emits real-time token events directly from LLM generation without artificial delays.
Safely extracts high-level progress status without exposing internal reasoning,
private chain-of-thought, or system prompts.
Persists multi-turn conversations in SQLite.
"""

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.agent.graph import create_research_graph
from app.agent.nodes import set_token_callback
from app.agent.state import ResearchState, create_initial_state
from app.storage.conversations import get_conversation_store

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


async def stream_research_events(
    question: str,
    max_iterations: int = 3,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream safe progress events, real-time answer tokens, and validated sources
    from the existing LangGraph execution over SSE.
    """
    clean_q = question.strip() if question else ""
    if not clean_q:
        yield _format_sse_event("error", {"message": "Question cannot be empty."})
        yield _format_sse_event("done", {})
        return

    start_time = time.time()
    store = get_conversation_store()

    # Multi-turn conversation management
    conv_id = store.create_conversation(
        title=clean_q[:60],
        conversation_id=conversation_id,
    )
    chat_history = store.get_recent_chat_history(conv_id, max_turns=5)

    # Save incoming user question to persistence store
    store.add_message(conversation_id=conv_id, role="user", content=clean_q)

    # Emit conversation ID to client
    yield _format_sse_event("conversation", {"conversation_id": conv_id})

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
    recorded_steps: List[Dict[str, Any]] = [
        {
            "step": "init",
            "title": "Understanding research question",
            "state": "completed",
            "details": "Initialized autonomous research workflow",
        }
    ]

    def run_graph_sync():
        try:
            # Register thread-local token streaming callback
            def on_token(delta: str):
                if delta:
                    event_queue.put(("token", delta))

            set_token_callback(on_token)

            graph = create_research_graph()
            initial_state = create_initial_state(
                question=clean_q,
                max_iterations=max_iterations,
                conversation_id=conv_id,
                chat_history=chat_history,
            )

            # Stream node updates from LangGraph
            for event in graph.stream(initial_state, stream_mode="updates"):
                event_queue.put(("node_update", event))

            event_queue.put((_SENTINEL, None))
        except Exception as ex:
            logger.exception("Error during LangGraph execution in worker thread")
            event_queue.put(("error", ex))
            event_queue.put((_SENTINEL, None))
        finally:
            set_token_callback(None)

    # Execute LangGraph in background worker thread to prevent blocking asyncio loop
    worker_thread = threading.Thread(target=run_graph_sync, daemon=True)
    worker_thread.start()

    answer_text = ""
    validated_sources: List[Dict[str, Any]] = []
    tokens_streamed_count = 0

    try:
        while True:
            # Check for events from worker thread without busy-waiting
            try:
                item_type, item_data = await asyncio.to_thread(event_queue.get, timeout=0.08)
            except queue.Empty:
                await asyncio.sleep(0.01)
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

            # Handle genuine streaming token directly from LLM
            if item_type == "token":
                tokens_streamed_count += 1
                yield _format_sse_event("token", {"text": item_data})
                continue

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
                        details = (
                            f"Generated {len(queries)} search query"
                            if len(queries) == 1
                            else f"Generated {len(queries)} search queries"
                        )
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

                    step_info = {
                        "step": node_name,
                        "title": config["title"],
                        "state": "completed",
                        "details": details,
                    }
                    recorded_steps.append(step_info)

                    # Emit completion status for current node
                    yield _format_sse_event("status", step_info)

                    # If model did not stream tokens for some reason (e.g. non-streaming fallback),
                    # emit the answer in batch so frontend always gets text
                    if node_name == "generate_answer" and tokens_streamed_count == 0 and answer_text:
                        yield _format_sse_event("token", {"text": answer_text})

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

                    # When citations are validated, emit sources event
                    if node_name == "validate_citations":
                        yield _format_sse_event(
                            "sources",
                            {"sources": validated_sources},
                        )

        # Save assistant answer, reasoning steps, and validated sources to SQLite store
        store.add_message(
            conversation_id=conv_id,
            role="assistant",
            content=answer_text,
            steps=recorded_steps,
            sources=validated_sources,
        )

        elapsed = round(time.time() - start_time, 2)
        yield _format_sse_event("done", {"elapsed_seconds": elapsed, "conversation_id": conv_id})

    except Exception as e:
        logger.exception(f"Unhandled error in research stream: {e}")
        yield _format_sse_event(
            "error",
            {"message": "Something went wrong while researching this question. Please try again."},
        )
        yield _format_sse_event("done", {})
