import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


def generate_point_id(chunk_id: str) -> str:
    """Generate a deterministic UUID string from a chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def ensure_collection(
        self,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        """
        Ensure the Qdrant collection exists with the exact vector dimension
        specified by the local embedding model.
        """
        try:
            collections_resp = self.client.get_collections()
            existing_names = [c.name for c in collections_resp.collections]

            if self.collection_name not in existing_names:
                logger.info(
                    f"Creating Qdrant collection '{self.collection_name}' (vector_size={vector_size}, distance={distance})"
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
            else:
                logger.debug(f"Qdrant collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection '{self.collection_name}': {e}")
            raise

    def delete_document_chunks(self, document_id: str) -> None:
        """
        Delete any existing chunks for a given document_id to guarantee idempotency.
        """
        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_condition),
            )
            logger.info(f"Purged previous chunks for document_id '{document_id}' in Qdrant")
        except Exception as e:
            logger.warning(f"Could not purge existing chunks for document_id '{document_id}': {e}")

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Upsert chunk vectors, payload metadata, and raw chunk text into Qdrant.
        """
        if not chunks:
            logger.warning("No chunks to store in Qdrant")
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
            )

        vector_size = len(embeddings[0])
        self.ensure_collection(vector_size=vector_size)

        # Delete pre-existing points for all documents in the batch to prevent orphaned chunks
        unique_doc_ids = {chunk.document_id for chunk in chunks}
        for doc_id in unique_doc_ids:
            self.delete_document_chunks(doc_id)

        points: List[models.PointStruct] = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = generate_point_id(chunk.chunk_id)

            payload: Dict[str, Any] = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "text": chunk.text,
                "url": chunk.url,
                "title": chunk.title,
                "domain": getattr(chunk, "domain", ""),
                "source_type": getattr(chunk, "source_type", "web"),
                "content_hash": getattr(chunk, "content_hash", ""),
                "crawled_at": getattr(chunk, "crawled_at", ""),
                "created_at": chunk.created_at,
                "embedding_model": settings.embedding_model,
                "metadata": chunk.metadata,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert points
        logger.info(f"Upserting {len(points)} points into Qdrant collection '{self.collection_name}'")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        query_vector: List[float],
        limit: int = 3,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest chunks given a query vector. Useful for verification and retrieval.
        """
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        try:
            # Standard search API
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                for hit in search_results
            ]
        except AttributeError:
            # Newer query_points API fallback if search is missing
            query_res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return [
                {
                    "id": point.id,
                    "score": getattr(point, "score", None),
                    "payload": point.payload,
                }
                for point in query_res.points
            ]


def store_chunks(
    chunks: List[DocumentChunk],
    embeddings: List[List[float]],
    collection_name: Optional[str] = None,
) -> int:
    """Convenience functional interface for storing chunks."""
    store = QdrantStore(collection_name=collection_name)
    return store.upsert_chunks(chunks=chunks, embeddings=embeddings)
