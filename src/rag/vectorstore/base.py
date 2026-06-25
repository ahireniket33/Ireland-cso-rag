"""Vector store interface and shared types."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SearchHit:
    chunk_id: str
    score: float            # cosine similarity in [-1, 1]
    text: str
    metadata: dict = field(default_factory=dict)
    title: str = ""


class VectorStore(abc.ABC):
    @abc.abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        metadatas: list[dict],
        titles: list[str],
    ) -> None: ...

    @abc.abstractmethod
    def query(self, embedding: np.ndarray, top_k: int) -> list[SearchHit]: ...

    @abc.abstractmethod
    def persist(self) -> None: ...

    @abc.abstractmethod
    def load(self) -> bool:
        """Load a persisted index. Returns True if one existed."""

    @abc.abstractmethod
    def count(self) -> int: ...
