"""Build the vector index from chunks.jsonl: embed -> add -> persist."""
from __future__ import annotations

from rag.config import Config
from rag.embeddings.factory import build_embedder
from rag.ingest.pipeline import load_chunks
from rag.logging_utils import get_logger
from rag.vectorstore.factory import build_store

log = get_logger(__name__)


def build_index(cfg: Config) -> int:
    chunks_file = cfg.path("chunks_file")
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"{chunks_file} not found. Run ingestion first (python run.py ingest)."
        )
    rows = load_chunks(chunks_file)
    if not rows:
        raise RuntimeError("No chunks to index.")

    embedder = build_embedder(cfg)
    store = build_store(cfg)

    texts = [r["text"] for r in rows]
    ids = [r["chunk_id"] for r in rows]
    titles = [r.get("title", "") for r in rows]
    metas = [r.get("metadata", {}) for r in rows]

    log.info("Embedding %d chunks with %s", len(texts), embedder.name)
    vectors = embedder.embed(texts)
    store.add(ids, vectors, texts, metas, titles)
    store.persist()
    log.info("INDEX DONE: %d chunks in '%s' store -> %s",
             store.count(), store.name if hasattr(store, "name") else "?",
             cfg.path("vector_dir"))
    return store.count()
