"""Zero-dependency numpy vector store (cosine). Persistable. Used as a light
fallback for tests/CI when FAISS/Chroma are not desired."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.vectorstore.base import SearchHit, VectorStore


class NumpyStore(VectorStore):
    name = "numpy"

    def __init__(self, persist_dir: Path) -> None:
        self.dir = Path(persist_dir)
        self._vecs: np.ndarray | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._meta: list[dict] = []
        self._titles: list[str] = []

    def add(self, ids, embeddings, texts, metadatas, titles):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._vecs = embeddings if self._vecs is None else np.vstack([self._vecs, embeddings])
        self._ids += list(ids)
        self._texts += list(texts)
        self._meta += list(metadatas)
        self._titles += list(titles)

    def query(self, embedding, top_k):
        if self._vecs is None or len(self._ids) == 0:
            return []
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        sims = self._vecs @ q  # vectors assumed L2-normalized
        order = np.argsort(-sims)[:top_k]
        return [
            SearchHit(
                chunk_id=self._ids[i], score=float(sims[i]), text=self._texts[i],
                metadata=self._meta[i], title=self._titles[i],
            )
            for i in order
        ]

    def persist(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "vectors.npy", self._vecs)
        (self.dir / "store.json").write_text(json.dumps({
            "ids": self._ids, "texts": self._texts,
            "meta": self._meta, "titles": self._titles,
        }), encoding="utf-8")

    def load(self):
        vp, sp = self.dir / "vectors.npy", self.dir / "store.json"
        if not (vp.exists() and sp.exists()):
            return False
        self._vecs = np.load(vp)
        d = json.loads(sp.read_text(encoding="utf-8"))
        self._ids, self._texts = d["ids"], d["texts"]
        self._meta, self._titles = d["meta"], d["titles"]
        return True

    def count(self):
        return len(self._ids)
