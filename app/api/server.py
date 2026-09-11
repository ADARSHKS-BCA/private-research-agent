"""
FastAPI Server for the Private Research Agent.

Provides endpoints for health checks and Server-Sent Events (SSE) research streaming.
"""

from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.adapter import stream_research_events
from app.config import settings

app = FastAPI(
    title="Private Research Agent API",
    description="Thin streaming API layer for the autonomous LangGraph research engine.",
    version="1.0.0",
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


@app.get("/api/health")
async def health_check():
    """Health check and configuration status."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
        "qdrant_collection": settings.qdrant_collection,
    }


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
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


def start():
    """Run server via uvicorn."""
    import uvicorn
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
