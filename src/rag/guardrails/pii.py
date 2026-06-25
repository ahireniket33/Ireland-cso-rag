"""PII detection & redaction (defence-in-depth on inputs and outputs)."""
from __future__ import annotations

import re

_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Irish PPS number: 7 digits + 1 or 2 letters
    "PPSN": re.compile(r"\b\d{7}[A-Wa-w]{1,2}\b"),
    "PHONE": re.compile(r"\b(?:\+?353|0)\s?\d(?:[\s-]?\d){7,9}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def detect_pii(text: str) -> list[str]:
    found = []
    for name, pat in _PATTERNS.items():
        if pat.search(text):
            found.append(name)
    return found


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    out = text
    for name, pat in _PATTERNS.items():
        if pat.search(out):
            found.append(name)
            out = pat.sub(f"[REDACTED_{name}]", out)
    return out, found
