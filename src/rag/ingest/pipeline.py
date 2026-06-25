"""End-to-end ingestion: download -> parse -> clean -> dedup -> chunk -> JSONL."""
from __future__ import annotations

import json
from pathlib import Path

from rag.config import Config
from rag.ingest.chunk import Chunk, chunk_documents
from rag.ingest.clean import Document, deduplicate, observations_to_documents
from rag.ingest.cso_client import CSOClient, CSOClientError, load_local
from rag.ingest.jsonstat import parse_dataset
from rag.logging_utils import get_logger

log = get_logger(__name__)


def _load_doc(cfg: Config, client: CSOClient, matrix: str) -> dict | None:
    """Try live API; on failure fall back to cached raw or bundled sample."""
    raw_dir = cfg.path("raw_dir")
    sample_dir = cfg.path("sample_dir")
    # Deterministic/offline mode: use only the bundled sample (for tests & CI),
    # never the live API, so results are reproducible regardless of network.
    if cfg.get("ingest", "offline", default=False):
        return load_local(matrix, sample_dir, raw_dir)
    try:
        doc = client.fetch(matrix)
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{matrix}.jsonstat.json").write_text(json.dumps(doc), encoding="utf-8")
        return doc
    except CSOClientError as exc:
        log.warning("Live fetch failed for %s (%s); trying local cache/sample", matrix, exc)
        return load_local(matrix, raw_dir, sample_dir)


def build_documents(cfg: Config) -> list[Document]:
    client = CSOClient(
        base_url=cfg.get("ingest", "base_url"),
        timeout_s=cfg.get("ingest", "request_timeout_s", default=30),
        retries=cfg.get("ingest", "request_retries", default=3),
        backoff_s=cfg.get("ingest", "request_backoff_s", default=2.0),
    )
    min_year = cfg.get("ingest", "min_year")
    all_docs: list[Document] = []
    for spec in cfg.datasets:
        doc = _load_doc(cfg, client, spec.code)
        if doc is None:
            log.error("No data available for %s (skipped)", spec.code)
            continue
        try:
            meta, observations = parse_dataset(doc)
        except (ValueError, KeyError) as exc:
            log.error("Malformed dataset %s rejected: %s", spec.code, exc)
            continue
        docs = observations_to_documents(meta, observations, min_year=min_year)
        log.info("matrix %s -> %d documents", spec.code, len(docs))
        all_docs.extend(docs)
    return deduplicate(all_docs)


def run_ingest(cfg: Config) -> Path:
    """Run ingestion and write chunks.jsonl. Returns the path."""
    docs = build_documents(cfg)
    if not docs:
        raise RuntimeError("Ingestion produced no documents (no data sources reachable).")

    chunks: list[Chunk] = chunk_documents(
        docs,
        strategy=cfg.get("chunk", "strategy", default="recursive"),
        size=cfg.get("chunk", "size", default=600),
        overlap=cfg.get("chunk", "overlap", default=100),
        min_chars=cfg.get("chunk", "min_chars", default=40),
    )

    out = cfg.path("chunks_file")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "text": c.text,
                "metadata": c.metadata,
            }) + "\n")
    log.info("INGEST DONE: %d documents -> %d chunks -> %s", len(docs), len(chunks), out)
    return out


def load_chunks(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
