"""Dependency-free deterministic hashing embedder.

Used for hermetic offline tests / CI (no HuggingFace download). It encodes
token unigrams+bigrams into a fixed-dimension vector via feature hashing with
sub-linear term weighting, then L2-normalizes. Not as strong as a neural model,
but stable, fast, and good enough to exercise the full RAG path deterministically.
"""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from rag.embeddings.base import Embedder, l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams


def _hash(token: str, dim: int) -> tuple[int, int]:
    h = hashlib.md5(token.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1 if h[4] & 1 else -1
    return idx, sign


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in _tokens(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                idx, sign = _hash(tok, self.dim)
                out[i, idx] += sign * (1.0 + math.log(c))
        return l2_normalize(out)
