from unittest.mock import MagicMock, patch
import pytest
from app.ingestion.chunker import DocumentChunk
from app.ingestion.embedder import LocalEmbedder


def test_embedder_dimension_and_vectors():
    # Mock backend to test LocalEmbedder logic safely in test suite
    with patch.object(LocalEmbedder, "_initialize_model") as mock_init:
        embedder = LocalEmbedder.__new__(LocalEmbedder)
        embedder.model_name = "BAAI/bge-m3"
        embedder._backend = "fastembed"
        embedder._dimension = None

        mock_fastembed_model = MagicMock()
        mock_embedding_1024 = MagicMock()
        mock_embedding_1024.tolist.return_value = [0.1] * 1024
        mock_fastembed_model.embed.return_value = [mock_embedding_1024]
        embedder._model = mock_fastembed_model

        vec = embedder.embed_text("Sample research text")
        assert len(vec) == 1024
        assert embedder.get_embedding_dimension() == 1024

        chunk = DocumentChunk(
            document_id="doc_1",
            chunk_id="doc_1_chunk_0",
            chunk_index=0,
            total_chunks=1,
            text="Sample research text",
            url="https://example.com",
            title="Sample",
            source_type="webpage",
            created_at="2026-09-08T00:00:00Z",
        )

        vectors = embedder.embed_chunks([chunk])
        assert len(vectors) == 1
        assert len(vectors[0]) == 1024
