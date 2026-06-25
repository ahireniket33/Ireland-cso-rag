"""Neural embedding backend (sentence-transformers). Default for production."""
from __future__ import annotations

import numpy as np

from rag.embeddings.base import Embedder, l2_normalize
from rag.logging_utils import get_logger

log = get_logger(__name__)


class SentenceTransformerEmbedder(Embedder):
    name = "sentence_transformers"

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        log.info("Loading embedding model %s", model)
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.normalize = normalize

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype(np.float32)
        return l2_normalize(vecs) if self.normalize else vecs
