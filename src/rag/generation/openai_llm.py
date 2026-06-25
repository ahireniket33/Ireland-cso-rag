"""OpenAI-compatible LLM generator (OpenAI, Together, Groq, Ollama, ...).

Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment. Never logs keys.
"""
from __future__ import annotations

import os

from rag.generation.base import (
    SYSTEM_PROMPT,
    GenerationResult,
    Generator,
    format_context,
)
from rag.logging_utils import get_logger
from rag.vectorstore.base import SearchHit

log = get_logger(__name__)


class OpenAIGenerator(Generator):
    name = "openai"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 timeout_s: int = 30) -> None:
        from openai import OpenAI  # lazy import
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "generation.backend=openai requires OPENAI_API_KEY in the environment."
            )
        base_url = os.getenv("OPENAI_BASE_URL") or None
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, question: str, hits: list[SearchHit]) -> GenerationResult:
        if not hits:
            return GenerationResult(answer="", used_context=False)
        context = format_context(hits)
        user = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above, and cite the CSO matrix code(s)."
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
        return GenerationResult(answer=answer, used_context=True)
