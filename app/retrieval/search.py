import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import settings

# Load environment variables
load_dotenv()

# Configuration
QDRANT_HOST = settings.qdrant_host
QDRANT_PORT = settings.qdrant_port
QDRANT_COLLECTION = settings.qdrant_collection
EMBEDDING_MODEL = settings.embedding_model

# Connect to Qdrant
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)

# Model helper for backwards compatibility and tests
_model = None


def get_model():
    """Retrieve embedding model using the unified local embedder."""
    global _model
    if _model is None:
        try:
            from app.ingestion.embedder import get_embedder
            embedder = get_embedder(EMBEDDING_MODEL)

            class _EmbedderAdapter:
                def __init__(self, emb):
                    self._emb = emb

                def encode(self, text, normalize_embeddings=True):
                    vec = self._emb.embed_text(text)
                    class _VectorList(list):
                        def tolist(self):
                            return self
                    return _VectorList(vec)

            _model = _EmbedderAdapter(embedder)
        except Exception:
            _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def search(query: str, top_k: int = 5):
    """
    Search Qdrant for chunks relevant to the user's query.
    Handles non-existent collection or connection issues gracefully.
    """
    model = get_model()

    # Convert query into vector
    encoded = model.encode(
        query,
        normalize_embeddings=True,
    )
    query_vector = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)

    try:
        try:
            collections_resp = client.get_collections()
            if hasattr(collections_resp, "collections") and collections_resp.collections:
                existing_names = [getattr(c, "name", None) for c in collections_resp.collections]
                existing_names = [n for n in existing_names if isinstance(n, str)]
                if existing_names and QDRANT_COLLECTION not in existing_names:
                    return []
        except Exception:
            pass

        # Search Qdrant
        try:
            results = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            return results.points if hasattr(results, "points") else results
        except AttributeError:
            results = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
            return results
    except Exception as e:
        print(f"[Warning] Retrieval search notice: {e}")
        return []


def main():
    query = input("\nEnter your question: ")

    print("\nSearching Qdrant...\n")

    results = search(query, top_k=5)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, start=1):
        payload = result.payload or {}

        print("=" * 70)
        print(f"RESULT {i}")
        print("=" * 70)

        print(f"Score: {result.score}")
        print(f"Title: {payload.get('title', 'N/A')}")
        print(f"URL: {payload.get('url', 'N/A')}")

        print("\nText:")
        print(payload.get("text", "N/A"))
        print()


if __name__ == "__main__":
    main()
