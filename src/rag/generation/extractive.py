"""Extractive generator: composes a grounded answer straight from retrieved
context. No API key, no cost, and faithful by construction (the answer text is
drawn from the context). Ideal default + free-tier deployment."""
from __future__ import annotations

import re

from rag.generation.base import GenerationResult, Generator
from rag.vectorstore.base import SearchHit

_SENT = re.compile(r"(?<=[.!?])\s+")
_STOP = {
    "the", "a", "an", "of", "in", "for", "to", "and", "was", "is", "are", "what",
    "how", "many", "much", "did", "do", "does", "on", "at", "by", "from", "with",
    "ireland", "irish", "value", "values", "annual",
}


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


class ExtractiveGenerator(Generator):
    name = "extractive"

    def __init__(self, max_context_chunks: int = 5) -> None:
        self.max_context_chunks = max_context_chunks

    def generate(self, question: str, hits: list[SearchHit]) -> GenerationResult:
        if not hits:
            return GenerationResult(answer="", used_context=False)

        qk = _keywords(question)
        years = set(re.findall(r"(?:19|20)\d{2}", question))

        # Score every sentence by: query/sentence keyword overlap + a year bonus
        # + a DOCUMENT-level topic bonus. The topic bonus anchors extraction to
        # the chunk whose subject matches the question (e.g. "inflation" ->
        # the CPI series), so we don't answer from a wrongly top-ranked chunk.
        scored: list[tuple[float, str, SearchHit]] = []
        for h in hits[: self.max_context_chunks]:
            doc_kw = _keywords(h.title + " " + h.text)
            topic_bonus = 2 * len(qk & doc_kw)
            for sent in _SENT.split(h.text):
                sent = sent.strip()
                if not sent:
                    continue
                sk = _keywords(sent)
                overlap = len(qk & sk) + topic_bonus
                if years and any(y in sent for y in years):
                    overlap += 3
                if overlap:
                    scored.append((overlap, sent, h))

        top_hit = hits[0]
        if not scored:
            # fall back to the single most relevant chunk's lead sentence
            lead = _SENT.split(top_hit.text)[0].strip()
            answer = lead
            cite_hit = top_hit
        else:
            scored.sort(key=lambda t: t[0], reverse=True)
            best_score, best_sent, cite_hit = scored[0]
            answer = best_sent

        primary = cite_hit.metadata.get('matrix', '?')
        cite = f"[Source: CSO {primary} — {cite_hit.title}]"
        return GenerationResult(answer=f"{answer} {cite}", used_context=True, primary_matrix=primary)
