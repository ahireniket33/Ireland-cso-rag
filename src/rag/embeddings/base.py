"""Embedding backend interface."""
from __future__ import annotations

import abc

import numpy as np


class Embedder(abc.ABC):
    """Common interface for all embedding backends."""

    name: str
    dim: int

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim)."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms
