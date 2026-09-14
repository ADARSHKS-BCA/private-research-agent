"""
FastAPI Server for the Private Research Agent.

Provides endpoints for:
- Health and readiness checks (/health, /api/health, /api/health/ready)
- Server-Sent Events (SSE) research streaming (/api/research/stream)
- Local Document Parsing and Vector Indexing (/api/documents/upload)
- Multi-Turn Conversation History Management (/api/conversations)
- Research Report Export (/api/research/{conversation_id}/export)
"""

import logging
from pathlib import Path
import shutil
import sys
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.adapter import stream_research_events
from app.config import settings
from app.export.exporter import export_as_json, export_as_markdown, export_as_pdf
from app.ingestion.doc_parser import ingest_document_file
from app.storage.conversations import get_conversation_store

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Private Research Agent API",
    description="Autonomous LangGraph research engine with local document ingestion and multi-turn persistence.",
    version="1.1.0",
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    question: str = Field(..., description="The user's research query", min_length=1)
    max_iterations: Optional[int] = Field(default=3, ge=1, le=5, description="Maximum research loops")
    conversation_id: Optional[str] = Field(default=None, description="Optional conversation session ID")


# =====================================================================
# Health & Diagnostics Endpoints
# =====================================================================
@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Liveness probe and system configuration status."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "qdrant_collection": settings.qdrant_collection,
    }


@app.get("/health/ready")
@app.get("/api/health/ready")
async def readiness_check():
    """Readiness probe verifying vector store and essential service connectivity."""
    checks = {
        "qdrant": "unknown",
        "llm_provider": settings.llm_provider,
        "database": "ok",
    }

    # Verify Qdrant connectivity
    try:
        from app.retrieval.search import get_client
        client = get_client()
        client.get_collections()
        checks["qdrant"] = "connected"
    except Exception as ex:
        checks["qdrant"] = f"unreachable ({type(ex).__name__})"

    # Verify SQLite conversation store
    try:
        store = get_conversation_store()
        store.list_conversations(limit=1)
        checks["database"] = "ready"
    except Exception as ex:
        checks["database"] = f"error ({type(ex).__name__})"

    all_ok = checks["database"] == "ready"
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }


# =====================================================================
# Research Streaming Endpoints (SSE)
# =====================================================================
@app.post("/api/research/stream")
async def stream_research_post(request: ResearchRequest):
    """
    Execute autonomous research and stream real-time events over SSE.
    """
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return StreamingResponse(
        stream_research_events(
            question=clean_question,
            max_iterations=request.max_iterations or 3,
            conversation_id=request.conversation_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


@app.get("/api/research/stream")
async def stream_research_get(
    q: str = Query(..., description="Research question"),
    max_iterations: int = Query(3, ge=1, le=5, description="Maximum research loops"),
    conversation_id: Optional[str] = Query(None, description="Optional conversation session ID"),
):
    """
    GET alternative for SSE research stream (useful for browser/EventSource testing).
    """
    clean_question = q.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return StreamingResponse(
        stream_research_events(
            question=clean_question,
            max_iterations=max_iterations,
            conversation_id=conversation_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


# =====================================================================
# Document Ingestion Endpoint
# =====================================================================
@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a local PDF, DOCX, TXT, or MD document.
    Parses pages, generates structure-aware chunks, embeds with BGE-M3,
    and indexes into the Qdrant knowledge base.
    """
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(allowed_extensions))}",
        )

    # Save to upload directory
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / filename

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        result = ingest_document_file(dest_path)
        return {
            "status": "success",
            "filename": filename,
            "document_id": result.get("document_id"),
            "document_type": result.get("document_type"),
            "total_pages": result.get("total_pages", 1),
            "chunks_indexed": result.get("chunks_indexed", 0),
            "message": f"Successfully parsed and indexed {result.get('chunks_indexed', 0)} chunks from {filename} into knowledge base.",
        }
    except Exception as e:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail=f"Document parsing/indexing failed: {e}")


# =====================================================================
# Multi-Turn Conversation History Endpoints
# =====================================================================
@app.get("/api/conversations")
async def list_conversations(limit: int = Query(50, ge=1, le=100)):
    """Retrieve list of past conversation sessions."""
    store = get_conversation_store()
    return {"conversations": store.list_conversations(limit=limit)}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Retrieve full message history and steps for a conversation session."""
    store = get_conversation_store()
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation session and all its messages."""
    store = get_conversation_store()
    deleted = store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success", "message": f"Conversation {conversation_id} deleted"}


# =====================================================================
# Research Report Export Endpoint
# =====================================================================
@app.get("/api/research/{conversation_id}/export")
async def export_research_report(
    conversation_id: str,
    format: str = Query("markdown", regex="^(markdown|json|pdf)$"),
):
    """
    Export research report for a conversation session.
    Supported formats: markdown (.md), json (.json), pdf (.pdf).
    """
    store = get_conversation_store()
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation session not found")

    messages = conv.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Conversation contains no messages to export")

    # Find last user question and assistant answer
    user_q = ""
    assistant_answer = ""
    sources = []

    for msg in reversed(messages):
        if msg.get("role") == "assistant" and not assistant_answer:
            assistant_answer = msg.get("content", "")
            sources = msg.get("sources") or []
        elif msg.get("role") == "user" and not user_q:
            user_q = msg.get("content", "")

    if not user_q and conv.get("title"):
        user_q = conv["title"]

    report_data = {
        "question": user_q,
        "answer": assistant_answer,
        "sources": sources,
        "timestamp": conv.get("updated_at"),
    }

    clean_filename = f"research_report_{conversation_id[:8]}"

    if format == "json":
        json_content = export_as_json(report_data)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{clean_filename}.json"'},
        )
    elif format == "pdf":
        pdf_bytes = export_as_pdf(report_data)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{clean_filename}.pdf"'},
        )
    else:  # markdown
        md_content = export_as_markdown(report_data)
        return Response(
            content=md_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{clean_filename}.md"'},
        )


def start():
    """Run server via uvicorn."""
    import uvicorn
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
