#!/usr/bin/env python3
"""Single reproducible entrypoint for the Ireland CSO RAG system.

Usage:
  python run.py pipeline                      # ingest + index (one command)
  python run.py ingest                        # download/clean/chunk only
  python run.py index                         # build vector index only
  python run.py query --question "..."        # ask a single question
  python run.py api [--host H --port P]       # serve the FastAPI app
  python run.py eval                          # run the RAG evaluation suite
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make src/ importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag.config import load_config  # noqa: E402
from rag.logging_utils import get_logger  # noqa: E402

log = get_logger("run")


def _cfg(args):
    return load_config(getattr(args, "config", None) or os.getenv("RAG_CONFIG", "config.yaml"))


def cmd_ingest(args):
    from rag.ingest.pipeline import run_ingest
    run_ingest(_cfg(args))


def cmd_index(args):
    from rag.indexer import build_index
    build_index(_cfg(args))


def cmd_pipeline(args):
    from rag.indexer import build_index
    from rag.ingest.pipeline import run_ingest
    cfg = _cfg(args)
    run_ingest(cfg)
    build_index(cfg)
    log.info("PIPELINE COMPLETE. Try: python run.py query --question \"What was Irish inflation in 2022?\"")


def cmd_query(args):
    from rag.pipeline import RAGPipeline
    pipe = RAGPipeline(_cfg(args))
    resp = pipe.answer(args.question)
    print(json.dumps(resp.to_dict(), indent=2, ensure_ascii=False))


def cmd_api(args):
    import uvicorn
    cfg = _cfg(args)
    host = args.host or cfg.get("api", "host", default="0.0.0.0")
    port = args.port or cfg.get("api", "port", default=8000)
    uvicorn.run("rag.api.app:app", host=host, port=int(port), log_level="info")


def cmd_eval(args):
    from rag.eval.evaluate import run_eval
    report = run_eval(_cfg(args))
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)


def main():
    parser = argparse.ArgumentParser(description="Ireland CSO RAG")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest").set_defaults(func=cmd_ingest)
    sub.add_parser("index").set_defaults(func=cmd_index)
    sub.add_parser("pipeline").set_defaults(func=cmd_pipeline)
    sub.add_parser("eval").set_defaults(func=cmd_eval)

    q = sub.add_parser("query")
    q.add_argument("--question", "-q", required=True)
    q.set_defaults(func=cmd_query)

    a = sub.add_parser("api")
    a.add_argument("--host", default=None)
    a.add_argument("--port", default=None)
    a.set_defaults(func=cmd_api)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
