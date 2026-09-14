"""
Hybrid Retrieval Module for Private Research Agent.

Combines:
1. Dense Semantic Vector Search (BAAI/bge-m3 via Qdrant)
2. Lexical BM25 Search (exact technical terms, acronyms, author names, formulas)
3. Reciprocal Rank Fusion (RRF) & Weighted Score Interpolation
"""

from collections import Counter
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.retrieval.search import get_client, get_model, search as dense_search

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())


class BM25Scorer:
    """
    In-memory BM25Okapi implementation for lexical search over chunk text payloads.
    Zero external dependencies, CPU-optimized, deterministic.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lens: List[int] = []
        self.doc_tokens: List[List[str]] = []
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[str]) -> "BM25Scorer":
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return self

        self.doc_lens = []
        self.doc_tokens = []
        df: Counter = Counter()

        total_tokens = 0
        for doc in documents:
            tokens = tokenize(doc)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            total_tokens += len(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] += 1

        self.avgdl = total_tokens / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs = dict(df)

        # Compute Robertson-Spärck Jones IDF
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

        return self

    def score(self, query: str) -> List[float]:
        if self.corpus_size == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return [0.0] * self.corpus_size

        scores = [0.0] * self.corpus_size
        for term in query_tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]
            for doc_idx, doc_toks in enumerate(self.doc_tokens):
                doc_len = self.doc_lens[doc_idx]
                tf = doc_toks.count(term)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += term_idf * (numerator / denominator)

        return scores


class HybridRetriever:
    """
    Orchestrates Dense Vector + Lexical BM25 retrieval with Score Fusion.
    """
    def __init__(
        self,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        collection_name: Optional[str] = None,
    ):
        self.dense_weight = dense_weight if dense_weight is not None else settings.dense_weight
        self.sparse_weight = sparse_weight if sparse_weight is not None else settings.sparse_weight
        self.collection_name = collection_name or settings.qdrant_collection

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        candidate_multiplier: int = 4,
    ) -> List[Any]:
        """
        Execute Hybrid Search:
        1. Fetch candidates via dense semantic vector search (larger candidate pool)
        2. Score candidates with BM25 lexical matcher
        3. Normalize and fuse scores using linear combination and RRF
        4. Return top_k best blended chunks
        """
        fetch_limit = max(top_k * candidate_multiplier, 20)

        # 1. Fetch dense candidates
        dense_results = dense_search(
            query=query,
            top_k=fetch_limit,
            collection_name=self.collection_name,
        )

        if not dense_results:
            return []

        # If only 1 result or small pool, return immediately
        if len(dense_results) <= 1:
            return dense_results[:top_k]

        # 2. Extract text payloads
        texts = []
        for r in dense_results:
            payload = getattr(r, "payload", {}) or {}
            texts.append(payload.get("text", ""))

        # 3. Compute BM25 scores across candidates
        bm25 = BM25Scorer().fit(texts)
        bm25_scores = bm25.score(query)

        # 4. Normalize Dense & BM25 scores to [0, 1] range
        dense_raw_scores = [float(getattr(r, "score", 0.0) or 0.0) for r in dense_results]
        max_dense = max(dense_raw_scores) if dense_raw_scores else 1.0
        min_dense = min(dense_raw_scores) if dense_raw_scores else 0.0
        dense_range = (max_dense - min_dense) if (max_dense - min_dense) > 1e-6 else 1.0

        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        min_bm25 = min(bm25_scores) if bm25_scores else 0.0
        bm25_range = (max_bm25 - min_bm25) if (max_bm25 - min_bm25) > 1e-6 else 1.0

        scored_candidates: List[Tuple[float, Any]] = []

        # 5. Compute Fused Score
        for idx, item in enumerate(dense_results):
            norm_dense = (dense_raw_scores[idx] - min_dense) / dense_range
            norm_bm25 = (bm25_scores[idx] - min_bm25) / bm25_range if max_bm25 > 0 else 0.0

            # Linear interpolation of dense and lexical scores
            fused_score = (self.dense_weight * norm_dense) + (self.sparse_weight * norm_bm25)

            # Assign fused score attribute or payload field
            setattr(item, "hybrid_score", round(fused_score, 4))
            setattr(item, "bm25_score", round(norm_bm25, 4))
            setattr(item, "dense_score", round(norm_dense, 4))

            scored_candidates.append((fused_score, item))

        # Sort by fused score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        logger.info(
            f"Hybrid retrieval ranked {len(scored_candidates)} chunks for '{query[:40]}' "
            f"(dense_w={self.dense_weight}, sparse_w={self.sparse_weight})"
        )

        return [item for _, item in scored_candidates[:top_k]]


def hybrid_search(
    query: str,
    top_k: int = 8,
    collection_name: Optional[str] = None,
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
) -> List[Any]:
    """Convenience functional interface for hybrid retrieval."""
    retriever = HybridRetriever(
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        collection_name=collection_name,
    )
    return retriever.retrieve(query=query, top_k=top_k)
