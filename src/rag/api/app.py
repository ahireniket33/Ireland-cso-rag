"""FastAPI app exposing /health and /query with logging, timeouts, error handling."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from rag import __version__
from rag.api.schemas import HealthResponse, QueryRequest, QueryResponse
from rag.config import load_config
from rag.logging_utils import get_logger
from rag.pipeline import RAGPipeline

log = get_logger("rag.api")

app = FastAPI(
    title="Ireland CSO RAG API",
    version=__version__,
    description="Grounded, cited Q&A over Irish CSO economic & census statistics.",
)

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the web UI (a dark, glassy front-end that calls /query)."""
    return FileResponse(_STATIC_DIR / "index.html")


_STATIC_DIR = Path(__file__).resolve().parent / "static"

_pipeline: RAGPipeline | None = None
_cfg = None


def get_pipeline() -> RAGPipeline:
    global _pipeline, _cfg
    if _pipeline is None:
        _cfg = load_config(os.getenv("RAG_CONFIG", "config.yaml"))
        log.info("Initialising RAG pipeline...")
        _pipeline = RAGPipeline(_cfg, load_index=True)
    return _pipeline


@app.on_event("startup")
async def _startup() -> None:
    # Warm up the pipeline; if no index is present (e.g. fresh container),
    # build it in-process so the service is self-healing on first boot.
    global _pipeline
    try:
        p = get_pipeline()
        if p.store.count() == 0:
            log.warning("No index found at startup \u2014 building it now...")
            from rag.indexer import build_index
            from rag.ingest.pipeline import run_ingest
            run_ingest(_cfg)
            build_index(_cfg)
            _pipeline = None          # force reload with the fresh index
            get_pipeline()
            log.info("Startup index build complete.")
    except Exception as exc:  # pragma: no cover
        log.error("Startup index build failed: %s", exc)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    p = get_pipeline()
    return HealthResponse(
        status="ok",
        indexed_chunks=p.store.count(),
        embeddings_backend=_cfg.get("embeddings", "backend"),
        vectorstore_backend=_cfg.get("vectorstore", "backend"),
        generation_backend=_cfg.get("generation", "backend"),
        version=__version__,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    p = get_pipeline()
    timeout_s = _cfg.get("api", "request_timeout_s", default=60)
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(p.answer, req.question), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out.") from None
    except Exception as exc:  # pragma: no cover
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
    return QueryResponse(**resp.to_dict())


@app.exception_handler(Exception)
async def _unhandled(_request, exc):  # pragma: no cover
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
