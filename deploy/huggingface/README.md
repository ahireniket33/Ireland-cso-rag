---
title: Ireland CSO RAG
emoji: 🇮🇪
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Ireland CSO RAG — Hugging Face Space (Docker SDK)

This Space serves the FastAPI app from the project root `Dockerfile`.

- Health: `GET /health`
- Query: `POST /query` with body `{"question": "..."}`
- Interactive docs: `/docs`

## How this Space is built
The Space uses the repository `Dockerfile`. On build it runs the ingestion +
indexing pipeline against the bundled CSO sample (and refreshes from the live
CSO PxStat API at startup when reachable). The default generation backend is
the **extractive** generator, so no API key or paid LLM is required.

To use a neural embedding model on the Space, keep
`embeddings.backend: sentence_transformers` in `config.yaml` (default) — the
`all-MiniLM-L6-v2` model downloads automatically from the Hugging Face Hub.
