"""Optional cross-encoder reranker."""
from __future__ import annotations

from rag.logging_utils import get_logger
from rag.vectorstore.base import SearchHit

log = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model: str, top_n: int = 5) -> None:
        from sentence_transformers import CrossEncoder  # lazy import
        log.info("Loading reranker %s", model)
        self._model = CrossEncoder(model)
        self.top_n = top_n

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return hits
        scores = self._model.predict([(query, h.text) for h in hits])
        for h, s in zip(hits, scores):
            h.score = float(s)
        ranked = sorted(hits, key=lambda h: h.score, reverse=True)
        return ranked[: self.top_n]
