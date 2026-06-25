"""Typed configuration loaded from config.yaml (no secrets in here)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetSpec:
    code: str
    title: str
    topic: str = ""


@dataclass
class Config:
    """Lightweight typed view over the YAML config.

    We keep the raw dict available via ``.raw`` for rarely-used keys while
    exposing the common, hot-path settings as typed attributes.
    """

    raw: dict[str, Any]
    root: Path

    # ---- convenience typed accessors ----
    @property
    def domain_keywords(self) -> list[str]:
        return [k.lower() for k in self.raw["project"]["domain_keywords"]]

    @property
    def datasets(self) -> list[DatasetSpec]:
        return [DatasetSpec(**d) for d in self.raw["ingest"]["datasets"]]

    def path(self, key: str) -> Path:
        """Resolve a path from the ``paths`` block relative to project root."""
        return (self.root / self.raw["paths"][key]).resolve()

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path or os.getenv("RAG_CONFIG", "config.yaml")).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=cfg_path.parent)
