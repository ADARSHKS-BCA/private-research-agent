from dataclasses import dataclass, asdict, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Citation:
    source_id: str  # e.g. "S1", "S2"
    source_index: int
    title: str
    url: str
    domain: str
    chunk_id: str
    document_id: str
    score: Optional[float]
    snippet: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Phrases indicating insufficient information
NO_INFO_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "cannot find enough information",
    "cannot be answered",
    "not mentioned in the provided",
    "no information provided",
    "provided sources do not contain",
    "sources do not mention",
]


def extract_citations(search_results: List[Any]) -> List[Citation]:
    """
    Extract structured citation records from Qdrant query results with [S1], [S2] source IDs.
    """
    citations: List[Citation] = []

    for i, res in enumerate(search_results, start=1):
        payload = getattr(res, "payload", {}) or {}
        score = getattr(res, "score", None)

        title = payload.get("title", "Untitled Document")
        url = payload.get("url", "N/A")
        domain = payload.get("domain", "")
        chunk_id = payload.get("chunk_id", f"chunk_{i}")
        document_id = payload.get("document_id", "")
        text = payload.get("text", "").strip()

        # Create concise snippet preview
        snippet = text[:250].replace("\n", " ") + ("..." if len(text) > 250 else "")

        citation = Citation(
            source_id=f"S{i}",
            source_index=i,
            title=title,
            url=url,
            domain=domain,
            chunk_id=chunk_id,
            document_id=document_id,
            score=round(score, 4) if score is not None else None,
            snippet=snippet,
            metadata=payload,
        )
        citations.append(citation)

    return citations


def format_prompt_context_with_sources(citations: List[Citation], max_text_len: int = 1200) -> str:
    """
    Build structured context for LLM prompt with explicit [S1], [S2] tags.
    """
    context_blocks = []

    for c in citations:
        text = c.metadata.get("text", "").strip()
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."

        block = (
            f"[{c.source_id}]\n"
            f"Title: {c.title}\n"
            f"Domain: {c.domain}\n"
            f"Content:\n{text}"
        )
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def validate_citations(
    answer_text: str,
    citations: List[Citation],
) -> Tuple[List[Citation], List[str]]:
    """
    Validate all citations cited in the answer:
    - Extracts all [S1], [S2] tags from the answer text.
    - Verifies they match existing source IDs in citations.
    - Rejects fabricated IDs like [S99].
    - Returns (valid_citations, invalid_source_ids).
    """
    if not answer_text or not citations:
        return [], []

    lower_ans = answer_text.lower()
    for phrase in NO_INFO_PHRASES:
        if phrase in lower_ans:
            return [], []

    # Map of valid source_id -> Citation object (e.g. "S1" -> Citation)
    valid_map: Dict[str, Citation] = {c.source_id.upper(): c for c in citations}

    # Find all [S1], [S2], [s1, s2], [S1][S2] tags in answer
    cited_ids_raw = re.findall(r"\[\s*(S\d+)\s*\]", answer_text, re.IGNORECASE)

    # Also handle [S1, S2] composite patterns
    composite_matches = re.findall(r"\[\s*S\d+(?:\s*,\s*S\d+)+\s*\]", answer_text, re.IGNORECASE)
    for comp in composite_matches:
        sub_ids = re.findall(r"S\d+", comp, re.IGNORECASE)
        cited_ids_raw.extend(sub_ids)

    seen_valid_ids: Set[str] = set()
    invalid_ids: List[str] = []

    for raw_id in cited_ids_raw:
        clean_id = raw_id.upper().strip()
        if clean_id in valid_map:
            seen_valid_ids.add(clean_id)
        else:
            if clean_id not in invalid_ids:
                invalid_ids.append(clean_id)

    # Order valid citations by original index
    valid_citations = [c for c in citations if c.source_id in seen_valid_ids]

    # If no specific [S#] tags were cited but answer contains real facts and no refusal, fallback to retrieved citations
    if not valid_citations and not invalid_ids and len(answer_text.strip()) > 30:
        valid_citations = citations

    return valid_citations, invalid_ids


def filter_used_citations(citations: List[Citation], answer_text: str) -> List[Citation]:
    """Alias for validate_citations returning only valid citations list."""
    valid_cites, _ = validate_citations(answer_text, citations)
    return valid_cites


def format_final_sources_block(citations: List[Citation]) -> str:
    """
    Format verified research bibliography with actual URLs.
    """
    if not citations:
        return ""

    lines = [
        "Sources:",
    ]

    seen_urls: Set[str] = set()
    for c in citations:
        if c.url in seen_urls:
            continue
        seen_urls.add(c.url)
        lines.append(f"[{c.source_id}] {c.title}")
        lines.append(f"     {c.url}")
        lines.append("")

    return "\n".join(lines).strip()


def format_citations_block(citations: List[Citation]) -> str:
    """Alias for format_final_sources_block for backward compatibility."""
    return format_final_sources_block(citations)


# Alias for backward compatibility
format_prompt_context = format_prompt_context_with_sources
