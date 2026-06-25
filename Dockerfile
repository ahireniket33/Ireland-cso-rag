# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    RAG_CONFIG=config.yaml \
    HF_HOME=/app/.hf_cache \
    PORT=8000

WORKDIR /app

# System deps (faiss/chromadb wheels are manylinux; build tools rarely needed).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index at image build using the bundled sample so the container is
# runnable offline; on first run the entrypoint refreshes from the live CSO API
# if reachable.
RUN python run.py pipeline || true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["python run.py pipeline || true; uvicorn rag.api.app:app --host 0.0.0.0 --port ${PORT}"]
