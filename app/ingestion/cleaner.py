import logging
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.ingestion.scraper import ScrapedDocument

logger = logging.getLogger(__name__)


@dataclass
class CleanedDocument:
    document_id: str
    url: str
    title: str
    markdown: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Patterns representing obvious web boilerplate/noise
BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\s*\[?\s*skip to (main )?content\s*\]?\(?.*?\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*accept (all )?cookies\s*\]?\(?.*?\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*(manage|reject|cookie) preferences\s*\]?\(?.*?\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*this website uses cookies.*$", re.IGNORECASE),
    re.compile(r"^\s*share this (article|post|page) on.*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*share on (twitter|facebook|linkedin|reddit|x)\s*\]?\(?.*?\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[\s*\]\(.*?\)\s*$"),  # Empty link anchors
    re.compile(r"^\s*!\[\s*\]\(.*?\)\s*$"),  # Empty image links with no alt text
]


class Cleaner:
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir or settings.data_processed_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """
        Rule-based text cleaning that normalizes markdown while preserving
        structural elements (headings, code blocks, lists, tables).
        """
        if not text:
            return ""

        # Step 1: Normalize unicode characters (NFKC)
        normalized = unicodedata.normalize("NFKC", text)

        # Step 2: Extract code blocks to protect them from regex modifications
        code_blocks: List[str] = []

        def _save_code_block(match):
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks.append(match.group(0))
            return placeholder

        # Protect fenced code blocks (```...```) and inline code (`...`)
        protected = re.sub(r"```[\s\S]*?```", _save_code_block, normalized)

        # Step 3: Process line by line
        lines = protected.splitlines()
        cleaned_lines: List[str] = []

        for line in lines:
            trimmed = line.rstrip()

            # Skip lines matching boilerplate noise
            is_noise = any(pattern.match(trimmed) for pattern in BOILERPLATE_LINE_PATTERNS)
            if is_noise:
                continue

            cleaned_lines.append(trimmed)

        content = "\n".join(cleaned_lines)

        # Step 4: Collapse 3 or more consecutive newlines into 2
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Step 5: Restore code blocks
        for i, code_block in enumerate(code_blocks):
            content = content.replace(f"__CODE_BLOCK_{i}__", code_block)

        return content.strip()

    def clean_document(self, doc: ScrapedDocument) -> CleanedDocument:
        """
        Clean the scraped document and save to data/processed/<document_id>.md.
        Ensures the raw document is never modified.
        """
        logger.info(f"Cleaning document: {doc.document_id} ({doc.url})")
        cleaned_markdown = self.clean_text(doc.markdown)

        if not cleaned_markdown:
            logger.warning(f"Cleaned document {doc.document_id} is empty, preserving raw content as fallback")
            cleaned_markdown = doc.markdown.strip()

        # Update metadata
        metadata = dict(doc.metadata)
        metadata["cleaned"] = True
        metadata["raw_char_count"] = len(doc.markdown)
        metadata["cleaned_char_count"] = len(cleaned_markdown)

        # Save to data/processed/<document_id>.md
        processed_file_path = self.save_dir / f"{doc.document_id}.md"
        processed_file_path.write_text(cleaned_markdown, encoding="utf-8")
        logger.info(f"Saved cleaned markdown to {processed_file_path}")

        return CleanedDocument(
            document_id=doc.document_id,
            url=doc.url,
            title=doc.title,
            markdown=cleaned_markdown,
            metadata=metadata,
            processed_file_path=str(processed_file_path),
        )


def clean_document(doc: ScrapedDocument) -> CleanedDocument:
    """Convenience functional interface for cleaning."""
    cleaner = Cleaner()
    return cleaner.clean_document(doc)
