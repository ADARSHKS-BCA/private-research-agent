import logging
from typing import List, Optional

from app.config import settings
from app.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Local embedding generator using open-source models (default: BAAI/bge-m3).
    Ensures zero external API network leakage.
    Supports FastEmbed (ONNX-optimized) and SentenceTransformers.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._backend = None
        self._dimension: Optional[int] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the local embedding model using the best available backend."""
        logger.info(f"Loading local embedding model: {self.model_name}")

        # Strategy 1: Try FastEmbed (fast, ONNX, CPU-optimized, natively supported by Qdrant)
        try:
            from fastembed import TextEmbedding
            # FastEmbed model name translation if needed
            self._model = TextEmbedding(model_name=self.model_name)
            self._backend = "fastembed"
            logger.info(f"Initialized FastEmbed with model {self.model_name}")
            return
        except Exception as e:
            logger.debug(f"FastEmbed initialization skipped/failed: {e}")

        # Strategy 2: Try SentenceTransformers (PyTorch-based)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._backend = "sentence_transformers"
            logger.info(f"Initialized SentenceTransformers with model {self.model_name}")
            return
        except Exception as e:
            logger.debug(f"SentenceTransformers initialization skipped/failed: {e}")

        # Strategy 3: Try Ollama local embeddings
        try:
            import ollama
            # Test pinging local ollama
            test_resp = ollama.embeddings(model=self.model_name, prompt="ping")
            if "embedding" in test_resp:
                self._model = ollama
                self._backend = "ollama"
                logger.info(f"Initialized Ollama embedding backend with model {self.model_name}")
                return
        except Exception as e:
            logger.debug(f"Ollama embedding backend skipped/failed: {e}")

        raise RuntimeError(
            f"Failed to initialize local embedding model '{self.model_name}'. "
            f"Please ensure 'fastembed' or 'sentence-transformers' is installed in the virtual environment."
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a single string."""
        if not text.strip():
            text = " "

        if self._backend == "fastembed":
            # FastEmbed embed() returns a generator of numpy arrays
            embeddings = list(self._model.embed([text]))
            vector = embeddings[0].tolist()
        elif self._backend == "sentence_transformers":
            vector = self._model.encode(text, normalize_embeddings=True).tolist()
        elif self._backend == "ollama":
            response = self._model.embeddings(model=self.model_name, prompt=text)
            vector = response["embedding"]
        else:
            raise RuntimeError(f"Unknown embedding backend: {self._backend}")

        if self._dimension is None:
            self._dimension = len(vector)
        return vector

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of strings."""
        if not texts:
            return []

        # Handle empty strings gracefully
        sanitized = [t if t.strip() else " " for t in texts]

        if self._backend == "fastembed":
            embeddings = list(self._model.embed(sanitized))
            vectors = [vec.tolist() for vec in embeddings]
        elif self._backend == "sentence_transformers":
            embeddings = self._model.encode(sanitized, normalize_embeddings=True)
            vectors = [vec.tolist() for vec in embeddings]
        elif self._backend == "ollama":
            vectors = [self.embed_text(t) for t in sanitized]
        else:
            raise RuntimeError(f"Unknown embedding backend: {self._backend}")

        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])

        return vectors

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """Generate vector embeddings for document chunks."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def get_embedding_dimension(self) -> int:
        """
        Dynamically determine the embedding dimension of the loaded model.
        """
        if self._dimension is not None:
            return self._dimension

        # Determine dimension with a quick sample embedding
        sample_vec = self.embed_text("dimension check")
        self._dimension = len(sample_vec)
        logger.info(f"Model {self.model_name} embedding dimension: {self._dimension}")
        return self._dimension


# Module-level singleton instance for reuse
_default_embedder: Optional[LocalEmbedder] = None


def get_embedder(model_name: Optional[str] = None) -> LocalEmbedder:
    """Get or create singleton LocalEmbedder instance."""
    global _default_embedder
    if _default_embedder is None or (model_name and _default_embedder.model_name != model_name):
        _default_embedder = LocalEmbedder(model_name=model_name)
    return _default_embedder


def embed_chunks(chunks: List[DocumentChunk], model_name: Optional[str] = None) -> List[List[float]]:
    """Convenience functional interface for chunk embedding."""
    embedder = get_embedder(model_name=model_name)
    return embedder.embed_chunks(chunks)
