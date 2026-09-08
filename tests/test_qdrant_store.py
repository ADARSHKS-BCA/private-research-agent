from unittest.mock import MagicMock
import pytest
from app.ingestion.chunker import DocumentChunk
from app.ingestion.qdrant_store import QdrantStore, generate_point_id


def test_generate_point_id_deterministic():
    id1 = generate_point_id("doc_123_chunk_0")
    id2 = generate_point_id("doc_123_chunk_0")
    id3 = generate_point_id("doc_123_chunk_1")
    assert id1 == id2
    assert id1 != id3


def test_qdrant_store_upsert_payload_and_idempotency():
    mock_client = MagicMock()
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_client.get_collections.return_value = mock_collections

    store = QdrantStore(host="localhost", port=6333, collection_name="test_collection")
    store._client = mock_client

    chunk = DocumentChunk(
        document_id="doc_rag_1",
        chunk_id="doc_rag_1_chunk_0",
        chunk_index=0,
        total_chunks=1,
        text="Dense retrieval techniques in modern LLMs.",
        url="https://arxiv.org/abs/2005.11401",
        title="RAG Paper",
        source_type="research_paper",
        created_at="2026-09-08T00:00:00Z",
        metadata={"token_estimate": 8},
    )
    vector = [0.05] * 1024

    upserted_count = store.upsert_chunks([chunk], [vector])

    assert upserted_count == 1
    # Collection creation should be called with vector size 1024
    mock_client.create_collection.assert_called_once()
    # Delete previous chunks must be called to guarantee idempotency
    mock_client.delete.assert_called_once()
    # Upsert must be called with PointStruct containing text in payload
    mock_client.upsert.assert_called_once()

    call_kwargs = mock_client.upsert.call_args[1]
    points = call_kwargs["points"]
    assert len(points) == 1
    point = points[0]
    assert point.payload["text"] == "Dense retrieval techniques in modern LLMs."
    assert point.payload["document_id"] == "doc_rag_1"
    assert point.payload["url"] == "https://arxiv.org/abs/2005.11401"
