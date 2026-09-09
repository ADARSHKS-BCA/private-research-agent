from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Union

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    document_id: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    text: str
    url: str
    title: str
    domain: str = ""
    source_type: str = "web"
    content_hash: str = ""
    crawled_at: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_token_count(text: str) -> int:
    """
    Estimate token count using tiktoken if available, with a fast fallback (~4 chars/token).
    """
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        # Fallback estimation: ~4 characters per token for English text
        return max(1, len(text) // 4)


class StructureAwareChunker:
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})")

    def _split_into_atomic_units(self, markdown_text: str) -> List[str]:
        """
        Split markdown into atomic semantic units:
        1. Major sections by headings (#, ##, ###)
        2. Paragraphs by double newlines (\n\n)
        3. Sentences if a single paragraph exceeds chunk_size
        """
        heading_pattern = r"(?=(?:\n|^)#{1,6}\s+)"
        raw_sections = re.split(heading_pattern, markdown_text)
        sections = [s.strip() for s in raw_sections if s.strip()]

        units: List[str] = []

        for section in sections:
            paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
            for para in paragraphs:
                para_tokens = estimate_token_count(para)
                if para_tokens <= self.chunk_size:
                    units.append(para)
                else:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current_sent_group: List[str] = []
                    current_sent_tokens = 0

                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        s_tokens = estimate_token_count(sent)
                        if current_sent_tokens + s_tokens > self.chunk_size and current_sent_group:
                            units.append(" ".join(current_sent_group))
                            current_sent_group = [sent]
                            current_sent_tokens = s_tokens
                        else:
                            current_sent_group.append(sent)
                            current_sent_tokens += s_tokens

                    if current_sent_group:
                        units.append(" ".join(current_sent_group))

        return units

    def chunk_document(self, doc: Any) -> List[DocumentChunk]:
        """
        Chunk a processed document into structure-aware, overlapping chunks.
        Accepts ProcessedDocument or CleanedDocument.
        """
        document_id = getattr(doc, "document_id", "unknown_doc")
        url = getattr(doc, "url", "")
        title = getattr(doc, "title", "Untitled")
        domain = getattr(doc, "domain", "")
        doc_metadata = getattr(doc, "metadata", {}) or {}
        source_type = getattr(doc, "source_type", None) or doc_metadata.get("source_type", "web")
        content_hash = getattr(doc, "content_hash", "")
        crawled_at = getattr(doc, "crawled_at", "")

        # Content field could be cleaned_markdown or markdown
        markdown_text = getattr(doc, "cleaned_markdown", None) or getattr(doc, "markdown", "")

        units = self._split_into_atomic_units(markdown_text)
        if not units:
            logger.warning(f"No content found to chunk in document {document_id}")
            return []

        chunks_text: List[str] = []
        current_units: List[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = estimate_token_count(unit)

            if current_tokens + unit_tokens > self.chunk_size and current_units:
                chunk_str = "\n\n".join(current_units)
                chunks_text.append(chunk_str)

                # Compute overlap for next chunk
                overlap_units: List[str] = []
                overlap_tokens = 0
                for prev_unit in reversed(current_units):
                    p_tok = estimate_token_count(prev_unit)
                    if overlap_tokens + p_tok <= self.chunk_overlap:
                        overlap_units.insert(0, prev_unit)
                        overlap_tokens += p_tok
                    else:
                        break

                current_units = overlap_units + [unit]
                current_tokens = overlap_tokens + unit_tokens
            else:
                current_units.append(unit)
                current_tokens += unit_tokens

        if current_units:
            chunks_text.append("\n\n".join(current_units))

        created_at = datetime.now(timezone.utc).isoformat()
        total_chunks = len(chunks_text)
        result: List[DocumentChunk] = []

        for idx, text in enumerate(chunks_text):
            chunk_id = f"{document_id}_chunk_{idx}"
            chunk_metadata = {
                "token_estimate": estimate_token_count(text),
                "char_count": len(text),
                **doc_metadata,
            }

            chunk = DocumentChunk(
                document_id=document_id,
                chunk_id=chunk_id,
                chunk_index=idx,
                total_chunks=total_chunks,
                text=text,
                url=url,
                title=title,
                domain=domain,
                source_type=source_type,
                content_hash=content_hash,
                crawled_at=crawled_at,
                created_at=created_at,
                metadata=chunk_metadata,
            )
            result.append(chunk)

        return result


def chunk_document(
    doc: Any,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[DocumentChunk]:
    """Convenience functional interface for chunking."""
    chunker = StructureAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_document(doc)


def chunk_documents(
    docs: List[Any],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[DocumentChunk]:
    """Chunk a list of documents and return all chunks."""
    all_chunks: List[DocumentChunk] = []
    chunker = StructureAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for doc in docs:
        all_chunks.extend(chunker.chunk_document(doc))
    return all_chunks
