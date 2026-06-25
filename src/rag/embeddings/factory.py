"""Build the configured embedding backend."""
from __future__ import annotations

from rag.config import Config
from rag.embeddings.base import Embedder


def build_embedder(cfg: Config) -> Embedder:
    backend = cfg.get("embeddings", "backend", default="sentence_transformers")
    if backend == "hashing":
        from rag.embeddings.hashing import HashingEmbedder

        return HashingEmbedder(dim=cfg.get("embeddings", "dim", default=384))
    if backend == "sentence_transformers":
        from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model=cfg.get("embeddings", "model"),
            batch_size=cfg.get("embeddings", "batch_size", default=64),
            normalize=cfg.get("embeddings", "normalize", default=True),
        )
    raise ValueError(f"Unknown embeddings backend: {backend}")
