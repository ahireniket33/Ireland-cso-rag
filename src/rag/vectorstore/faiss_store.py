"""FAISS vector store (cosine via inner product on normalized vectors)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.vectorstore.base import SearchHit, VectorStore


class FaissStore(VectorStore):
    name = "faiss"

    def __init__(self, persist_dir: Path, distance: str = "cosine") -> None:
        self.dir = Path(persist_dir)
        self.distance = distance
        self._index = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._meta: list[dict] = []
        self._titles: list[str] = []

    def _new_index(self, dim: int):
        import faiss  # lazy
        return faiss.IndexFlatIP(dim)  # inner product == cosine on normalized

    def add(self, ids, embeddings, texts, metadatas, titles):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self._index is None:
            self._index = self._new_index(embeddings.shape[1])
        self._index.add(embeddings)
        self._ids += list(ids)
        self._texts += list(texts)
        self._meta += list(metadatas)
        self._titles += list(titles)

    def query(self, embedding, top_k):
        if self._index is None or self._index.ntotal == 0:
            return []
        q = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        scores, idxs = self._index.search(q, min(top_k, self._index.ntotal))
        hits: list[SearchHit] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            hits.append(SearchHit(
                chunk_id=self._ids[i], score=float(score), text=self._texts[i],
                metadata=self._meta[i], title=self._titles[i],
            ))
        return hits

    def persist(self):
        import faiss
        self.dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.dir / "index.faiss"))
        (self.dir / "store.json").write_text(json.dumps({
            "ids": self._ids, "texts": self._texts,
            "meta": self._meta, "titles": self._titles,
        }), encoding="utf-8")

    def load(self):
        import faiss
        ip, sp = self.dir / "index.faiss", self.dir / "store.json"
        if not (ip.exists() and sp.exists()):
            return False
        self._index = faiss.read_index(str(ip))
        d = json.loads(sp.read_text(encoding="utf-8"))
        self._ids, self._texts = d["ids"], d["texts"]
        self._meta, self._titles = d["meta"], d["titles"]
        return True

    def count(self):
        return 0 if self._index is None else self._index.ntotal
