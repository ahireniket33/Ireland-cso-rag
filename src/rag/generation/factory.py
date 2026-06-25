"""Build the configured generator."""
from __future__ import annotations

from rag.config import Config
from rag.generation.base import Generator


def build_generator(cfg: Config) -> Generator:
    backend = cfg.get("generation", "backend", default="extractive")
    if backend == "extractive":
        from rag.generation.extractive import ExtractiveGenerator
        return ExtractiveGenerator(
            max_context_chunks=cfg.get("generation", "max_context_chunks", default=5)
        )
    if backend == "openai":
        from rag.generation.openai_llm import OpenAIGenerator
        oc = cfg.get("generation", "openai", default={})
        return OpenAIGenerator(
            model=oc.get("model", "gpt-4o-mini"),
            temperature=cfg.get("generation", "temperature", default=0.0),
            max_tokens=cfg.get("generation", "max_tokens", default=512),
            timeout_s=oc.get("request_timeout_s", 30),
        )
    raise ValueError(f"Unknown generation backend: {backend}")
