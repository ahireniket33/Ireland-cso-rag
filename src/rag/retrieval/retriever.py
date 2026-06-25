"""Retriever: embed query -> vector search -> threshold -> optional rerank."""
from __future__ import annotations

from rag.config import Config
from rag.embeddings.base import Embedder
from rag.logging_utils import get_logger
from rag.vectorstore.base import SearchHit, VectorStore

log = get_logger(__name__)


class Retriever:
    def __init__(self, cfg: Config, embedder: Embedder, store: VectorStore) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.top_k = cfg.get("retrieval", "top_k", default=5)
        self.threshold = cfg.get("retrieval", "similarity_threshold", default=0.3)
        self._reranker = None
        if cfg.get("retrieval", "rerank", "enabled", default=False):
            from rag.retrieval.rerank import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(
                model=cfg.get("retrieval", "rerank", "model"),
                top_n=cfg.get("retrieval", "rerank", "top_n", default=5),
            )

    def retrieve(self, query: str) -> list[SearchHit]:
        """Return hits above the similarity threshold (anti-hallucination gate)."""
        qvec = self.embedder.embed_one(query)
        # Over-fetch when reranking so the reranker has candidates to reorder.
        k = max(self.top_k, self.cfg.get("retrieval", "rerank", "top_n", default=self.top_k)) \
            if self._reranker else self.top_k
        hits = self.store.query(qvec, top_k=k)

        if self._reranker:
            hits = self._reranker.rerank(query, hits)
            # rerank scores aren't cosine; keep ordering, skip cosine threshold.
            kept = hits[: self.top_k]
        else:
            kept = [h for h in hits if h.score >= self.threshold][: self.top_k]

        log.info("retrieved %d/%d hits above threshold %.2f for query=%r",
                 len(kept), len(hits), self.threshold, query[:60])
        return kept
