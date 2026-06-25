"""Generation interface and shared prompt scaffolding."""
from __future__ import annotations

import abc
from dataclasses import dataclass

from rag.vectorstore.base import SearchHit


@dataclass
class GenerationResult:
    answer: str
    used_context: bool
    primary_matrix: str | None = None


SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about Irish economic and "
    "census statistics from the Central Statistics Office (CSO). "
    "Answer ONLY using the provided context. If the context does not contain "
    "the answer, reply exactly: "
    "\"I don't have enough information to answer that from the Irish CSO statistics I have.\" "
    "Never invent numbers. Cite the source dataset (matrix code) for any figure."
)


def format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        cite = h.metadata.get("matrix", "?")
        blocks.append(f"[{i}] (source: CSO {cite}) {h.text}")
    return "\n".join(blocks)


class Generator(abc.ABC):
    name: str

    @abc.abstractmethod
    def generate(self, question: str, hits: list[SearchHit]) -> GenerationResult: ...
