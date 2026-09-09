from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    """
    Typed state object for the LangGraph Autonomous Research Agent.
    
    Fields:
        question: The user's original research question.
        search_queries: Generated sub-queries or search terms (1-3 queries per iteration).
        search_results: Discovered URLs and snippets from Firecrawl search.
        scraped_documents: Scraped raw web documents (ScrapedWebDocument instances or dicts).
        cleaned_documents: Boilerplate-stripped documents (ProcessedDocument instances or dicts).
        chunks: Structure-aware chunk objects (DocumentChunk instances or dicts).
        retrieved_documents: Most relevant evidence chunks retrieved from Qdrant.
        evidence_sufficient: Decision flag whether retrieved evidence answers the question.
        research_iteration: Current iteration counter for bounded search loops (1-indexed).
        max_iterations: Hard ceiling on research iterations to prevent infinite loops.
        answer: Generated factual synthesis citing [S1], [S2] source markers.
        sources: Verified citation metadata mapped to legitimate URLs.
        errors: List of non-fatal error notices recorded during graph execution.
    """
    question: str
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    scraped_documents: List[Any]
    cleaned_documents: List[Any]
    chunks: List[Any]
    retrieved_documents: List[Any]
    evidence_sufficient: bool
    research_iteration: int
    max_iterations: int
    answer: str
    sources: List[Dict[str, Any]]
    errors: List[str]


def create_initial_state(
    question: str,
    max_iterations: int = 3,
) -> ResearchState:
    """
    Helper function to initialize a clean ResearchState.
    """
    clean_q = question.strip() if question else ""
    return {
        "question": clean_q,
        "search_queries": [],
        "search_results": [],
        "scraped_documents": [],
        "cleaned_documents": [],
        "chunks": [],
        "retrieved_documents": [],
        "evidence_sufficient": False,
        "research_iteration": 1,
        "max_iterations": max_iterations,
        "answer": "",
        "sources": [],
        "errors": [],
    }
