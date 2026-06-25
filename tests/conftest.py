"""Shared pytest fixtures. Hermetic: hashing embedder + numpy store + sample data."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("RAG_LOG_LEVEL", "ERROR")

from rag.config import load_config  # noqa: E402
from rag.indexer import build_index  # noqa: E402
from rag.ingest.pipeline import run_ingest  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

CONFIG = "config.test.yaml"


@pytest.fixture(scope="session")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="session")
def built_index(cfg):
    # Build a fresh index from the bundled sample once per test session.
    vec_dir = cfg.path("vector_dir")
    if vec_dir.exists():
        shutil.rmtree(vec_dir, ignore_errors=True)
    run_ingest(cfg)
    build_index(cfg)
    return cfg


@pytest.fixture(scope="session")
def pipeline(built_index):
    return RAGPipeline(built_index, load_index=True)


@pytest.fixture(scope="session")
def sample_doc(cfg):
    import json
    p = cfg.path("sample_dir") / "CPM01.jsonstat.json"
    return json.loads(Path(p).read_text(encoding="utf-8"))
