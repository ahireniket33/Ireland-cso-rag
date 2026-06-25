"""Chroma vector store (local persistent client)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rag.vectorstore.base import SearchHit, VectorStore


class ChromaStore(VectorStore):
    name = "chroma"

    def __init__(self, persist_dir: Path, collection: str = "cso_chunks",
                 distance: str = "cosine") -> None:
        self.dir = Path(persist_dir)
        self.collection_name = collection
        self.distance = distance
        self._client = None
        self._col = None

    def _ensure(self):
        if self._col is not None:
            return
        import chromadb  # lazy
        from chromadb.config import Settings
        self.dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.dir), settings=Settings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": self.distance},
        )

    def add(self, ids, embeddings, texts, metadatas, titles):
        self._ensure()
        metas = []
        for m, t in zip(metadatas, titles):
            mm = {k: ("" if v is None else v) for k, v in m.items()}
            mm["title"] = t
            metas.append(mm)
        self._col.add(
            ids=list(ids),
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            documents=list(texts),
            metadatas=metas,
        )

    def query(self, embedding, top_k):
        self._ensure()
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=min(top_k, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            meta = dict(meta)
            title = meta.pop("title", "")
            # cosine distance -> similarity
            hits.append(SearchHit(chunk_id=cid, score=1.0 - float(dist),
                                  text=doc, metadata=meta, title=title))
        return hits

    def persist(self):
        # PersistentClient writes through automatically.
        self._ensure()

    def load(self):
        self._ensure()
        return self._col.count() > 0

    def count(self):
        self._ensure()
        return self._col.count()
