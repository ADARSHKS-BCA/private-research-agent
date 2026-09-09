"""
CLI Runner for the LangGraph Autonomous Research Agent.

Usage:
    python -m app.agent.runner
    python -m app.agent.runner --question "What are the latest approaches to agentic RAG?"
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.graph import create_research_graph
from app.agent.state import ResearchState, create_initial_state


def run_agent_research(
    question: str,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Execute the LangGraph autonomous research workflow for a given question.
    """
    clean_q = question.strip()
    if not clean_q:
        raise ValueError("Research question cannot be empty.")

    start_time = time.time()

    print("\n" + "=" * 70)
    print("  LANGGRAPH AUTONOMOUS RESEARCH AGENT")
    print("=" * 70)
    print(f"Research Question: \"{clean_q}\"\n")

    graph = create_research_graph()
    initial_state = create_initial_state(question=clean_q, max_iterations=max_iterations)

    final_state = graph.invoke(initial_state)

    elapsed = round(time.time() - start_time, 2)
    print(f"\nWorkflow finished in {elapsed}s.")

    # Format outputs
    answer = final_state.get("answer", "").strip()
    sources = final_state.get("sources", [])

    print("\n" + "=" * 50)
    print("FINAL ANSWER")
    print("=" * 50)
    print()
    print(answer if answer else "No answer could be generated from the available sources.")
    print()

    print("=" * 50)
    print("SOURCES")
    print("=" * 50)
    print()

    if sources:
        seen_urls = set()
        for idx, s in enumerate(sources, start=1):
            url = s.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            source_id = s.get("source_id", f"S{idx}")
            title = s.get("title", "Untitled")
            print(f"[{source_id}] {title}")
            print(f"{url}\n")
    else:
        print("No citations available.\n")

    return {
        "question": clean_q,
        "answer": answer,
        "sources": sources,
        "search_queries": final_state.get("search_queries", []),
        "search_results": final_state.get("search_results", []),
        "scraped_documents": len(final_state.get("scraped_documents", [])),
        "chunks": len(final_state.get("chunks", [])),
        "retrieved_documents": len(final_state.get("retrieved_documents", [])),
        "evidence_sufficient": final_state.get("evidence_sufficient", False),
        "research_iteration": final_state.get("research_iteration", 1),
        "elapsed_seconds": elapsed,
        "errors": final_state.get("errors", []),
    }


def main():
    parser = argparse.ArgumentParser(description="LangGraph Autonomous Research Agent CLI")
    parser.add_argument("question", nargs="?", type=str, help="Research question to investigate", default=None)
    parser.add_argument("--question", "-q", dest="question_flag", type=str, help="Research question flag", default=None)
    parser.add_argument("--max-iterations", type=int, help="Maximum search loops (default: 3)", default=3)
    args = parser.parse_args()

    target_q = args.question_flag or args.question

    if not target_q:
        try:
            target_q = input("Enter your research question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    if not target_q:
        print("Error: No research question provided.")
        return

    try:
        run_agent_research(question=target_q, max_iterations=args.max_iterations)
    except Exception as e:
        print(f"\n[Error] Agent execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
