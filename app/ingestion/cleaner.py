"""
Canonical Cleaner re-exports for backward compatibility.
All document cleaning implementation has been unified into app.processing.cleaner.
"""

from app.processing.cleaner import (  # noqa: F401
    CleanedDocument,
    Cleaner,
    ProcessedDocument,
    calculate_content_hash,
    clean_document,
    clean_markdown,
    generate_document_id,
    process_scraped_document,
    process_scraped_documents,
)

__all__ = [
    "CleanedDocument",
    "Cleaner",
    "ProcessedDocument",
    "calculate_content_hash",
    "clean_document",
    "clean_markdown",
    "generate_document_id",
    "process_scraped_document",
    "process_scraped_documents",
]
