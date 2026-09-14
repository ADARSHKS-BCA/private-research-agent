"""
CPU-Friendly Reranking Layer for Private Research Agent.

Re-ranks the candidate evidence chunks from hybrid retrieval using a Cross-Encoder
(or FlashRank) to ensure maximum relevance before context is passed to the LLM.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

_reranker_instance = None


class CrossEncoderReranker:
    """
    Lightweight CPU cross-encoder reranker with graceful fallback.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.reranker_model or "ms-marco-MiniLM-L-6-v2"
        self._model = None
        self._backend = None
        self._init_model()

    def _init_model(self):
        # Strategy 1: FlashRank (lightweight, ONNX CPU-optimized)
        try:
            from flashrank import Ranker
            self._model = Ranker(model_name=self.model_name)
            self._backend = "flashrank"
            logger.info(f"Initialized FlashRank reranker with model {self.model_name}")
            return
        except Exception:
            pass

        # Strategy 2: SentenceTransformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
            # Prefix model name if needed
            full_name = self.model_name
            if not full_name.startswith("cross-encoder/"):
                full_name = f"cross-encoder/{self.model_name}"
            self._model = CrossEncoder(full_name)
            self._backend = "cross_encoder"
            logger.info(f"Initialized CrossEncoder reranker with model {full_name}")
            return
        except Exception as e:
            logger.debug(f"CrossEncoder initialization skipped/failed: {e}")

        logger.info("Reranker running in passthrough mode (no cross-encoder loaded).")

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: int = 8,
    ) -> List[Any]:
        """
        Score (query, chunk_text) pairs and return top_k reranked items.
        """
        if not candidates:
            return []

        if len(candidates) <= 1 or not self._model:
            return candidates[:top_k]

        # Extract text from candidate payloads
        pairs = []
        indices = []
        for i, cand in enumerate(candidates):
            payload = getattr(cand, "payload", {}) or {}
            text = payload.get("text", "")
            if text:
                pairs.append([query, text[:1000]])
                indices.append(i)

        if not pairs:
            return candidates[:top_k]

        try:
            if self._backend == "flashrank":
                from flashrank import RerankRequest
                passages = [{"id": i, "text": p[1]} for i, p in enumerate(pairs)]
                req = RerankRequest(query=query, passages=passages)
                results = self._model.rerank(req)
                ranked_order = [candidates[indices[r["id"]]] for r in results]
                for r in results:
                    orig_cand = candidates[indices[r["id"]]]
                    setattr(orig_cand, "rerank_score", round(float(r["score"]), 4))
                return ranked_order[:top_k]

            elif self._backend == "cross_encoder":
                scores = self._model.predict(pairs)
                scored = []
                for score, orig_idx in zip(scores, indices):
                    cand = candidates[orig_idx]
                    setattr(cand, "rerank_score", round(float(score), 4))
                    scored.append((float(score), cand))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [c for _, c in scored[:top_k]]
        except Exception as ex:
            logger.warning(f"Reranking error: {ex}. Falling back to candidate order.")
            return candidates[:top_k]

        return candidates[:top_k]


def get_reranker() -> CrossEncoderReranker:
    """Singleton getter for reranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


def rerank_evidence(
    query: str,
    candidates: List[Any],
    top_k: Optional[int] = None,
) -> List[Any]:
    """
    Rerank candidate evidence chunks.
    Respects settings.reranker_enabled flag.
    """
    target_k = top_k or settings.reranker_top_k or 8

    if not settings.reranker_enabled:
        return candidates[:target_k]

    reranker = get_reranker()
    return reranker.rerank(query=query, candidates=candidates, top_k=target_k)
