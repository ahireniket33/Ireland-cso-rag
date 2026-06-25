"""CSO PxStat open-data API client (JSON-stat 2.0), with retries + timeouts.

Network note: in restricted environments cso.ie may be unreachable. The
ingestion pipeline falls back to bundled fixtures in ``data/sample/`` or any
previously cached file in ``data/raw/`` so the pipeline stays reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.logging_utils import get_logger

log = get_logger(__name__)


class CSOClientError(RuntimeError):
    pass


class CSOClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: int = 30,
        retries: int = 3,
        backoff_s: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def _url(self, matrix: str) -> str:
        return f"{self.base_url}/{matrix}/JSON-stat/2.0/en"

    def fetch(self, matrix: str) -> dict[str, Any]:
        """Fetch a dataset with bounded retries/backoff. Raises on failure."""

        @retry(
            stop=stop_after_attempt(self.retries),
            wait=wait_exponential(multiplier=self.backoff_s, min=self.backoff_s, max=30),
            retry=retry_if_exception_type((requests.RequestException,)),
            reraise=True,
        )
        def _do() -> dict[str, Any]:
            log.info("Fetching CSO matrix %s", matrix)
            resp = self._session.get(self._url(matrix), timeout=self.timeout_s)
            resp.raise_for_status()
            return resp.json()

        try:
            return _do()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise CSOClientError(f"Failed to fetch {matrix}: {exc}") from exc

    def fetch_to_file(self, matrix: str, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = self.fetch(matrix)
        out_path = out_dir / f"{matrix}.jsonstat.json"
        out_path.write_text(json.dumps(doc), encoding="utf-8")
        log.info("Saved %s (%d bytes)", out_path.name, out_path.stat().st_size)
        return out_path


def load_local(matrix: str, *dirs: Path) -> dict[str, Any] | None:
    """Load a cached/bundled JSON-stat file for ``matrix`` from given dirs."""
    for d in dirs:
        for name in (f"{matrix}.jsonstat.json", f"{matrix}.json"):
            p = d / name
            if p.exists():
                log.info("Loaded cached %s from %s", matrix, p)
                return json.loads(p.read_text(encoding="utf-8"))
    return None
