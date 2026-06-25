"""RAG orchestrator: guardrails -> retrieval -> generation -> grounding check.

This is the single entry into the RAG system. It enforces the anti-hallucination
contract: off-domain/injection queries are refused; low-confidence retrieval is
refused; ungrounded answers are suppressed; valid answers carry citations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag.config import Config
from rag.embeddings.factory import build_embedder
from rag.generation.factory import build_generator
from rag.guardrails.input_guards import InputGuard
from rag.guardrails.output_guards import OutputGuard
from rag.logging_utils import get_logger
from rag.retrieval.retriever import Retriever
from rag.vectorstore.factory import build_store

log = get_logger(__name__)


@dataclass
class Citation:
    matrix: str
    title: str
    url: str
    source: str = ""
    score: float = 0.0


@dataclass
class RAGResponse:
    answer: str
    refused: bool
    reason: str = ""
    citations: list[Citation] = field(default_factory=list)
    faithfulness: float = 1.0
    num_retrieved: int = 0
    flags: list[str] = field(default_factory=list)
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "refused": self.refused,
            "reason": self.reason,
            "citations": [c.__dict__ for c in self.citations],
            "faithfulness": self.faithfulness,
            "num_retrieved": self.num_retrieved,
            "flags": self.flags,
            "latency_ms": self.latency_ms,
        }


class RAGPipeline:
    def __init__(self, cfg: Config, load_index: bool = True) -> None:
        self.cfg = cfg
        self.refusal = cfg.get("guardrails", "refusal_message")
        self.input_guard = InputGuard(cfg)
        self.output_guard = OutputGuard(cfg)
        self.embedder = build_embedder(cfg)
        self.store = build_store(cfg)
        if load_index and not self.store.load():
            log.warning("No persisted index found; retrieval will return nothing. "
                        "Run `python run.py pipeline` first.")
        self.retriever = Retriever(cfg, self.embedder, self.store)
        self.generator = build_generator(cfg)

    def _refuse(self, reason: str, t0: float, msg: str | None = None,
                flags: list[str] | None = None) -> RAGResponse:
        return RAGResponse(
            answer=msg or self.refusal, refused=True, reason=reason,
            faithfulness=0.0, flags=flags or [],
            latency_ms=int((time.time() - t0) * 1000),
        )

    def answer(self, query: str) -> RAGResponse:
        t0 = time.time()

        # 1) Input guardrails
        decision = self.input_guard.check(query)
        if not decision.ok:
            msg = {
                "off_domain": "That question is outside this assistant's domain "
                              "(Irish CSO economic & census statistics).",
                "prompt_injection_detected": "Your request was blocked by the input "
                              "safety guard.",
                "empty_query": "Please enter a question.",
                "query_too_long": "Your question is too long.",
            }.get(decision.reason, self.refusal)
            log.info("input guard refused: %s", decision.reason)
            return self._refuse(decision.reason, t0, msg, flags=decision.pii_found)

        clean_query = decision.cleaned_query

        # 2) Retrieval (with confidence threshold)
        hits = self.retriever.retrieve(clean_query)
        if not hits:
            return self._refuse("no_relevant_context", t0,
                                flags=decision.pii_found)

        # 3) Generation (grounded)
        gen = self.generator.generate(clean_query, hits)
        if not gen.used_context or not gen.answer.strip():
            return self._refuse("empty_generation", t0)

        # Detect explicit model refusal
        if "don't have enough information" in gen.answer.lower():
            return self._refuse("model_refused", t0, gen.answer)

        # 4) Output grounding / faithfulness guard
        grounding = self.output_guard.check(gen.answer, hits)
        flags = list(decision.pii_found) + grounding.flags
        if not grounding.grounded:
            log.info("output guard suppressed ungrounded answer: %s", grounding.flags)
            return self._refuse("ungrounded_answer", t0,
                                self.refusal, flags=flags)

        # 5) Citations (answer-source dataset first)
        primary = gen.primary_matrix
        ordered = sorted(hits, key=lambda h: 0 if h.metadata.get('matrix') == primary else 1)
        seen = set()
        citations: list[Citation] = []
        for h in ordered:
            m = h.metadata.get("matrix", "?")
            if m in seen:
                continue
            seen.add(m)
            citations.append(Citation(
                matrix=m, title=h.title,
                url=h.metadata.get("url", ""),
                source=h.metadata.get("source", ""),
                score=round(h.score, 3),
            ))

        return RAGResponse(
            answer=grounding.answer, refused=False, reason="ok",
            citations=citations, faithfulness=grounding.faithfulness,
            num_retrieved=len(hits), flags=flags,
            latency_ms=int((time.time() - t0) * 1000),
        )
