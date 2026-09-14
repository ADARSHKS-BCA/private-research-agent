from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set
import unicodedata
from urllib.parse import urlparse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.web.scraper import ScrapedWebDocument

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    document_id: str
    url: str
    title: str
    cleaned_markdown: str
    raw_markdown: str = ""
    domain: str = ""
    source_type: str = "web"
    content_hash: str = ""
    crawled_at: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_file_path: Optional[str] = None
    processed_file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def markdown(self) -> str:
        """Alias for backward compatibility with CleanedDocument."""
        return self.cleaned_markdown

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias for backward compatibility
CleanedDocument = ProcessedDocument


# Web boilerplate patterns
BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\s*\[?\s*skip to (main )?(article|content)\s*\]?\(?.*?\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*accept (all )?cookies\s*\]?\(?.*?\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*(manage|reject|cookie) preferences\s*\]?\(?.*?\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*this website uses cookies.*$", re.IGNORECASE),
    re.compile(r"^\s*share this (article|post|page) on.*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*share on (twitter|facebook|linkedin|reddit|x)\s*\]?\(?.*?\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[\s*\]\(.*?\)\s*$"),  # Empty link anchors
    re.compile(r"^\s*!\[\s*\]\(.*?\)\s*$"),  # Empty image links with no alt text
]


def generate_document_id(url: str) -> str:
    """
    Generate a deterministic, unique document ID based on normalized URL.
    """
    parsed = urlparse(url.strip())
    normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    url_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"doc_{url_hash}"


def calculate_content_hash(text: str) -> str:
    """
    Generate SHA-256 hash of cleaned text for exact duplicate detection.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_markdown(raw_text: str) -> str:
    """
    Rule-based cleaning:
    - Normalizes unicode characters (NFKC)
    - Protects code blocks from regex modifications
    - Removes web boilerplate lines (cookies, skip navigation, share buttons)
    - Preserves headings, lists, tables, code blocks, technical text
    - Collapses excessive blank lines (max 2)
    """
    if not raw_text:
        return ""

    # Step 1: Normalize unicode characters (NFKC)
    normalized = unicodedata.normalize("NFKC", raw_text)

    # Step 2: Protect code blocks
    code_blocks: List[str] = []

    def _save_code_block(match):
        placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
        code_blocks.append(match.group(0))
        return placeholder

    protected = re.sub(r"```[\s\S]*?```", _save_code_block, normalized)

    # Step 3: Filter lines
    lines = protected.splitlines()
    cleaned_lines: List[str] = []

    for line in lines:
        trimmed = line.rstrip()
        is_noise = any(pattern.match(trimmed) for pattern in BOILERPLATE_LINE_PATTERNS)
        if is_noise:
            continue
        cleaned_lines.append(trimmed)

    content = "\n".join(cleaned_lines)

    # Step 4: Collapse 3+ consecutive newlines to 2
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Step 5: Restore code blocks
    for i, code_block in enumerate(code_blocks):
        content = content.replace(f"__CODE_BLOCK_{i}__", code_block)

    return content.strip()


def process_scraped_document(
    scraped_doc: Any,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> Optional[ProcessedDocument]:
    """
    Process a single scraped web document: clean, generate metadata, save files.
    Returns None if document is empty or scraping failed.
    """
    is_success = getattr(scraped_doc, "success", True)
    raw_content = getattr(scraped_doc, "markdown", "") or ""
    if not is_success or not raw_content.strip():
        return None

    save_raw = raw_dir or settings.data_raw_dir
    save_processed = processed_dir or settings.data_processed_dir
    save_raw.mkdir(parents=True, exist_ok=True)
    save_processed.mkdir(parents=True, exist_ok=True)

    url = getattr(scraped_doc, "url", "")
    title = getattr(scraped_doc, "title", "Untitled")
    domain = getattr(scraped_doc, "domain", "")
    description = getattr(scraped_doc, "description", "")
    crawled_at = getattr(scraped_doc, "crawled_at", "")
    doc_meta = getattr(scraped_doc, "metadata", {}) or {}

    document_id = getattr(scraped_doc, "document_id", None) or generate_document_id(url)
    cleaned_text = clean_markdown(raw_content)

    if not cleaned_text:
        return None

    content_hash = calculate_content_hash(cleaned_text)

    # Save raw document if not already saved
    raw_file_path = getattr(scraped_doc, "raw_file_path", None)
    if not raw_file_path:
        raw_out = save_raw / f"{document_id}.md"
        raw_out.write_text(raw_content, encoding="utf-8")
        raw_file_path = str(raw_out)

    # Save processed document
    processed_file_path = save_processed / f"{document_id}.md"
    processed_file_path.write_text(cleaned_text, encoding="utf-8")

    created_at = datetime.now(timezone.utc).isoformat()
    crawled_at = crawled_at or created_at

    metadata = {
        **doc_meta,
        "raw_char_count": len(raw_content),
        "cleaned_char_count": len(cleaned_text),
        "description": description,
    }

    return ProcessedDocument(
        document_id=document_id,
        url=url,
        title=title,
        cleaned_markdown=cleaned_text,
        raw_markdown=raw_content,
        domain=domain,
        source_type=getattr(scraped_doc, "source_type", "web"),
        content_hash=content_hash,
        crawled_at=crawled_at,
        created_at=created_at,
        raw_file_path=raw_file_path,
        processed_file_path=str(processed_file_path),
        metadata=metadata,
    )


def process_scraped_documents(
    scraped_docs: List[Any],
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> List[ProcessedDocument]:
    """
    Process multiple scraped documents with duplicate URL and content hash filtering.
    """
    processed: List[ProcessedDocument] = []
    seen_urls: Set[str] = set()
    seen_hashes: Set[str] = set()

    for doc in scraped_docs:
        success = getattr(doc, "success", True)
        url = getattr(doc, "url", "")
        if not success or not url:
            continue

        if url.lower() in seen_urls:
            logger.info(f"Skipping duplicate URL: {url}")
            continue

        processed_doc = process_scraped_document(doc, raw_dir=raw_dir, processed_dir=processed_dir)
        if not processed_doc:
            continue

        if processed_doc.content_hash in seen_hashes:
            logger.info(f"Skipping duplicate content hash for: {url}")
            continue

        seen_urls.add(url.lower())
        seen_hashes.add(processed_doc.content_hash)
        processed.append(processed_doc)

    return processed


class Cleaner:
    """
    Canonical Cleaner class interface for pipeline integration and backward compatibility.
    """
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir or settings.data_processed_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        return clean_markdown(text)

    def clean_document(self, doc: Any) -> ProcessedDocument:
        res = process_scraped_document(doc, processed_dir=self.save_dir)
        if res is None:
            # Fallback if empty
            raw = getattr(doc, "markdown", "")
            return ProcessedDocument(
                document_id=getattr(doc, "document_id", "doc_fallback"),
                url=getattr(doc, "url", ""),
                title=getattr(doc, "title", "Untitled"),
                cleaned_markdown=raw,
                raw_markdown=raw,
                domain=getattr(doc, "domain", ""),
            )
        return res


def clean_document(doc: Any) -> ProcessedDocument:
    cleaner = Cleaner()
    return cleaner.clean_document(doc)
