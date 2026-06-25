"""Build the configured vector store backend."""
from __future__ import annotations

from rag.config import Config
from rag.vectorstore.base import VectorStore


def build_store(cfg: Config) -> VectorStore:
    backend = cfg.get("vectorstore", "backend", default="chroma")
    persist_dir = cfg.path("vector_dir")
    distance = cfg.get("vectorstore", "distance", default="cosine")
    if backend == "faiss":
        from rag.vectorstore.faiss_store import FaissStore
        return FaissStore(persist_dir, distance=distance)
    if backend == "numpy":
        from rag.vectorstore.numpy_store import NumpyStore
        return NumpyStore(persist_dir)
    if backend == "chroma":
        from rag.vectorstore.chroma_store import ChromaStore
        return ChromaStore(
            persist_dir,
            collection=cfg.get("vectorstore", "collection", default="cso_chunks"),
            distance=distance,
        )
    raise ValueError(f"Unknown vectorstore backend: {backend}")
