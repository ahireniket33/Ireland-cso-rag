"""Input guardrails: validation, off-domain & prompt-injection blocking, PII."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import Config
from rag.guardrails.pii import redact_pii

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(system|previous|above)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|in)\b", re.I),
    re.compile(r"\b(system|developer)\s*prompt\b", re.I),
    re.compile(r"\bjailbreak\b|\bDAN\b", re.I),
    re.compile(r"reveal|print|show\s+(me\s+)?(your\s+)?(prompt|instructions|system)", re.I),
    re.compile(r"pretend\s+to\s+be|act\s+as\s+if", re.I),
]


@dataclass
class GuardDecision:
    ok: bool
    reason: str = ""
    cleaned_query: str = ""
    pii_found: list[str] = None  # type: ignore

    def __post_init__(self):
        if self.pii_found is None:
            self.pii_found = []


class InputGuard:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.max_chars = cfg.get("guardrails", "max_query_chars", default=500)
        self.keywords = cfg.domain_keywords
        self.enable_pii = cfg.get("guardrails", "enable_pii_filter", default=True)
        self.enabled = cfg.get("guardrails", "enable_input_guard", default=True)

    def check(self, query: str) -> GuardDecision:
        q = (query or "").strip()
        if not q:
            return GuardDecision(ok=False, reason="empty_query")
        if len(q) > self.max_chars:
            return GuardDecision(ok=False, reason="query_too_long")

        if not self.enabled:
            return GuardDecision(ok=True, cleaned_query=q)

        for pat in _INJECTION_PATTERNS:
            if pat.search(q):
                return GuardDecision(ok=False, reason="prompt_injection_detected")

        pii_found: list[str] = []
        if self.enable_pii:
            q, pii_found = redact_pii(q)

        if not self._on_domain(q):
            return GuardDecision(ok=False, reason="off_domain", pii_found=pii_found)

        return GuardDecision(ok=True, cleaned_query=q, pii_found=pii_found)

    def _on_domain(self, q: str) -> bool:
        low = q.lower()
        return any(kw in low for kw in self.keywords)
