from app.rag.citations import (
    Citation,
    extract_citations,
    filter_used_citations,
    format_citations_block,
    format_final_sources_block,
    format_prompt_context,
    format_prompt_context_with_sources,
    validate_citations,
)


def generate_answer(query: str, top_k: int = 3, stream: bool = True):
    """Lazy import to avoid runpy warnings when executing answer module directly."""
    from app.rag.answer import generate_answer as _generate_answer
    return _generate_answer(query=query, top_k=top_k, stream=stream)


__all__ = [
    "Citation",
    "extract_citations",
    "filter_used_citations",
    "format_citations_block",
    "format_final_sources_block",
    "format_prompt_context",
    "format_prompt_context_with_sources",
    "validate_citations",
    "generate_answer",
]
