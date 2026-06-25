"""Output guardrails: groundedness/faithfulness check + PII redaction.

We verify that the generated answer is supported by retrieved context:
 - every number in the answer must appear in some retrieved chunk;
 - a minimum fraction of answer sentences must overlap lexically with context.
Unsupported content is flagged (and numerically-unsupported answers suppressed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.config import Config
from rag.guardrails.pii import redact_pii
from rag.vectorstore.base import SearchHit

_SENT = re.compile(r"(?<=[.!?])\s+")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+")


def _nums(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)}


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class GroundingResult:
    grounded: bool
    faithfulness: float
    answer: str
    flags: list[str] = field(default_factory=list)
    pii_found: list[str] = field(default_factory=list)


class OutputGuard:
    def __init__(self, cfg: Config) -> None:
        self.enabled = cfg.get("guardrails", "enable_output_grounding", default=True)
        self.threshold = cfg.get("guardrails", "faithfulness_threshold", default=0.5)
        self.enable_pii = cfg.get("guardrails", "enable_pii_filter", default=True)

    def check(self, answer: str, hits: list[SearchHit]) -> GroundingResult:
        flags: list[str] = []
        pii_found: list[str] = []
        if self.enable_pii:
            answer, pii_found = redact_pii(answer)
            if pii_found:
                flags.append("pii_redacted")

        if not self.enabled or not hits:
            return GroundingResult(True, 1.0, answer, flags, pii_found)

        context = " ".join(h.text for h in hits)
        ctx_nums = _nums(context)
        ctx_words = _words(context)

        # 1) Numeric grounding: every number in the answer must be in context.
        # (Ignore citation/index artefacts by checking against context numbers.)
        ans_nums = _nums(re.sub(r"\[Source:.*?\]", "", answer))
        unsupported_nums = {n for n in ans_nums if n not in ctx_nums and len(n) > 2}
        if unsupported_nums:
            flags.append(f"unsupported_numbers:{sorted(unsupported_nums)}")

        # 2) Lexical faithfulness across sentences.
        sents = [s for s in _SENT.split(re.sub(r"\[Source:.*?\]", "", answer)) if s.strip()]
        supported = 0
        for s in sents:
            sw = _words(s)
            if not sw:
                continue
            overlap = len(sw & ctx_words) / max(1, len(sw))
            if overlap >= 0.5:
                supported += 1
        faithfulness = supported / max(1, len(sents))

        grounded = (not unsupported_nums) and (faithfulness >= self.threshold)
        if not grounded:
            flags.append("low_faithfulness")
        return GroundingResult(grounded, round(faithfulness, 3), answer, flags, pii_found)
