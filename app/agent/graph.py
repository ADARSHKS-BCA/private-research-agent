"""
LangGraph StateGraph Definition for the Autonomous Research Agent.

Orchestrates the research loop with conditional routing:
START
  ↓
plan_research
  ↓
search_web
  ↓
scrape_sources
  ↓
process_documents
  ↓
index_documents
  ↓
retrieve_evidence
  ↓
evaluate_evidence
  ├────── (if insufficient & iteration < max) ───► plan_research (loop)
  └────── (if sufficient or max iterations)   ───► generate_answer
                                                        ↓
                                                 validate_citations
                                                        ↓
                                                       END
"""

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    evaluate_evidence,
    generate_answer,
    index_documents,
    plan_research,
    process_documents,
    retrieve_evidence,
    scrape_sources,
    search_web_node,
    validate_citations,
)
from app.agent.state import ResearchState, create_initial_state


def should_continue(state: ResearchState) -> str:
    """
    Conditional routing function:
    - If evidence is sufficient or max iterations reached -> proceed to generate_answer
    - If evidence is insufficient and under max iterations -> loop back to plan_research
    """
    is_sufficient = state.get("evidence_sufficient", False)
    iteration = state.get("research_iteration", 1)
    max_iterations = state.get("max_iterations", 3)

    if is_sufficient or iteration >= max_iterations:
        return "generate_answer"
    return "plan_research"


def create_research_graph():
    """
    Construct, wire, and compile the LangGraph Autonomous Research Workflow.
    """
    workflow = StateGraph(ResearchState)

    # 1. Register Nodes
    workflow.add_node("plan_research", plan_research)
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("scrape_sources", scrape_sources)
    workflow.add_node("process_documents", process_documents)
    workflow.add_node("index_documents", index_documents)
    workflow.add_node("retrieve_evidence", retrieve_evidence)
    workflow.add_node("evaluate_evidence", evaluate_evidence)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("validate_citations", validate_citations)

    # 2. Register Linear Pipeline Edges
    workflow.add_edge(START, "plan_research")
    workflow.add_edge("plan_research", "search_web")
    workflow.add_edge("search_web", "scrape_sources")
    workflow.add_edge("scrape_sources", "process_documents")
    workflow.add_edge("process_documents", "index_documents")
    workflow.add_edge("index_documents", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "evaluate_evidence")

    # 3. Register Conditional Edge for Autonomous Research Loop
    workflow.add_conditional_edges(
        "evaluate_evidence",
        should_continue,
        {
            "plan_research": "plan_research",
            "generate_answer": "generate_answer",
        },
    )

    # 4. Final Grounding & Citation Validation
    workflow.add_edge("generate_answer", "validate_citations")
    workflow.add_edge("validate_citations", END)

    return workflow.compile()
