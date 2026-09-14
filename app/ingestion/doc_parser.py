"""
Local Document Parser for Private Research Agent.

Supports:
- PDF (.pdf) with page-level extraction
- Microsoft Word (.docx)
- Plain Text (.txt)
- Markdown (.md)

Extracts clean markdown and structured metadata for chunking, embedding, and Qdrant indexing.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Union
import zipfile
import xml.etree.ElementTree as ET

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.ingestion.chunker import DocumentChunk, chunk_document, estimate_token_count
from app.ingestion.embedder import get_embedder
from app.ingestion.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    title: str
    markdown: str
    document_type: str
    total_pages: int = 1
    page_texts: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_file_document_id(filename: str, content: bytes) -> str:
    """Generate a stable document ID from filename and content SHA-256."""
    content_hash = hashlib.sha256(content).hexdigest()[:16]
    clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(filename).stem)[:24]
    return f"file_{clean_name}_{content_hash}"


def parse_pdf(file_path: Path) -> ParsedDocument:
    """Extract text page-by-page from a PDF document."""
    content_bytes = file_path.read_bytes()
    doc_id = generate_file_document_id(file_path.name, content_bytes)
    page_texts: Dict[int, str] = {}
    title = file_path.stem.replace("_", " ").replace("-", " ").title()

    # Try pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        num_pages = len(reader.pages)
        full_text_pages = []

        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                page_texts[idx] = text
                full_text_pages.append(f"## Page {idx}\n\n{text}")

        # Extract title from PDF metadata if available
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title.strip()

        combined_markdown = f"# {title}\n\n" + "\n\n".join(full_text_pages)

        return ParsedDocument(
            document_id=doc_id,
            filename=file_path.name,
            title=title,
            markdown=combined_markdown,
            document_type="pdf",
            total_pages=num_pages,
            page_texts=page_texts,
            metadata={
                "source": file_path.name,
                "document_type": "pdf",
                "total_pages": num_pages,
                "file_size_bytes": len(content_bytes),
            },
        )
    except ImportError:
        logger.warning("pypdf is not installed. Attempting basic PDF text extraction...")
    except Exception as ex:
        logger.warning(f"pypdf extraction error: {ex}. Attempting fallback...")

    # Fallback string extraction for PDFs if pypdf unavailable
    raw_str = content_bytes.decode("latin-1", errors="ignore")
    text_pieces = re.findall(r"\(([^\(\)]+)\)\s*T[jJ]", raw_str)
    extracted = " ".join(text_pieces) if text_pieces else ""

    return ParsedDocument(
        document_id=doc_id,
        filename=file_path.name,
        title=title,
        markdown=f"# {title}\n\n{extracted}",
        document_type="pdf",
        total_pages=1,
        page_texts={1: extracted},
        metadata={"source": file_path.name, "document_type": "pdf"},
    )


def parse_docx(file_path: Path) -> ParsedDocument:
    """Extract paragraphs and headings from a Word .docx document."""
    content_bytes = file_path.read_bytes()
    doc_id = generate_file_document_id(file_path.name, content_bytes)
    title = file_path.stem.replace("_", " ").replace("-", " ").title()

    # Try python-docx
    try:
        import docx
        doc = docx.Document(str(file_path))
        markdown_lines = [f"# {title}\n"]

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            # Check style for headings
            style_name = getattr(p.style, "name", "").lower()
            if "heading 1" in style_name:
                markdown_lines.append(f"## {text}\n")
            elif "heading 2" in style_name:
                markdown_lines.append(f"### {text}\n")
            elif "heading 3" in style_name:
                markdown_lines.append(f"#### {text}\n")
            else:
                markdown_lines.append(f"{text}\n")

        combined = "\n".join(markdown_lines)
        return ParsedDocument(
            document_id=doc_id,
            filename=file_path.name,
            title=title,
            markdown=combined,
            document_type="docx",
            total_pages=1,
            metadata={"source": file_path.name, "document_type": "docx"},
        )
    except Exception:
        pass

    # Fallback: parse document.xml inside .docx zip directly (no python-docx needed!)
    try:
        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for p in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                texts = [
                    t.text for t in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                    if t.text
                ]
                if texts:
                    paragraphs.append("".join(texts))
            combined = f"# {title}\n\n" + "\n\n".join(paragraphs)
            return ParsedDocument(
                document_id=doc_id,
                filename=file_path.name,
                title=title,
                markdown=combined,
                document_type="docx",
                total_pages=1,
                metadata={"source": file_path.name, "document_type": "docx"},
            )
    except Exception as ex:
        logger.error(f"Docx fallback parsing failed: {ex}")

    return ParsedDocument(
        document_id=doc_id,
        filename=file_path.name,
        title=title,
        markdown=f"# {title}\n\n(Could not extract docx content)",
        document_type="docx",
    )


def parse_text_or_markdown(file_path: Path, doc_type: str = "txt") -> ParsedDocument:
    """Parse plain text or markdown file."""
    content_bytes = file_path.read_bytes()
    doc_id = generate_file_document_id(file_path.name, content_bytes)
    title = file_path.stem.replace("_", " ").replace("-", " ").title()

    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = content_bytes.decode("latin-1", errors="ignore")

    # If first line has heading, use as title
    lines = text.splitlines()
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            title = s.lstrip("#").strip()
            break

    return ParsedDocument(
        document_id=doc_id,
        filename=file_path.name,
        title=title,
        markdown=text,
        document_type=doc_type,
        total_pages=1,
        metadata={"source": file_path.name, "document_type": doc_type},
    )


def parse_file(file_path: Union[str, Path]) -> ParsedDocument:
    """
    Detect document format and parse into a structured ParsedDocument.
    Supports .pdf, .docx, .txt, .md.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    elif suffix == ".docx":
        return parse_docx(path)
    elif suffix == ".md":
        return parse_text_or_markdown(path, doc_type="markdown")
    elif suffix in (".txt", ".text", ".rst"):
        return parse_text_or_markdown(path, doc_type="txt")
    else:
        # Fallback to plain text reader
        return parse_text_or_markdown(path, doc_type="other")


def ingest_document_file(
    file_path: Union[str, Path],
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete end-to-end ingestion pipeline for local documents:
    Parse -> Structure-Aware Chunk -> Embed (BGE-M3) -> Store in Qdrant.
    """
    path = Path(file_path)
    parsed = parse_file(path)

    if not parsed.markdown.strip():
        raise ValueError(f"No text content could be extracted from {path.name}")

    target_collection = collection_name or settings.qdrant_collection

    # Create chunks using StructureAwareChunker
    raw_chunks = chunk_document(parsed)
    if not raw_chunks:
        raise ValueError(f"No chunks generated from document {path.name}")

    # Enrich chunk metadata with page numbers and document type
    enhanced_chunks: List[DocumentChunk] = []
    for c in raw_chunks:
        page_num = 1
        # Try finding page reference from chunk text
        m = re.search(r"##\s*Page\s*(\d+)", c.text, re.IGNORECASE)
        if m:
            page_num = int(m.group(1))

        meta = {
            **c.metadata,
            "source": parsed.filename,
            "filename": parsed.filename,
            "page": page_num,
            "document_type": parsed.document_type,
            "source_type": "file",
        }

        enhanced = DocumentChunk(
            document_id=c.document_id,
            chunk_id=c.chunk_id,
            chunk_index=c.chunk_index,
            total_chunks=c.total_chunks,
            text=c.text,
            url=f"file://{parsed.filename}",
            title=c.title,
            domain="local_document",
            source_type="file",
            content_hash=parsed.document_id,
            created_at=c.created_at,
            metadata=meta,
        )
        enhanced_chunks.append(enhanced)

    # Embed with BGE-M3
    embedder = get_embedder(model_name=settings.embedding_model)
    embeddings = embedder.embed_chunks(enhanced_chunks)

    # Store in Qdrant
    store = QdrantStore(collection_name=target_collection)
    stored_count = store.upsert_chunks(chunks=enhanced_chunks, embeddings=embeddings)

    return {
        "status": "success",
        "document_id": parsed.document_id,
        "filename": parsed.filename,
        "document_type": parsed.document_type,
        "total_pages": parsed.total_pages,
        "chunks_indexed": stored_count,
        "collection": target_collection,
    }
