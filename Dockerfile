# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Non-root user (Hugging Face Spaces runs as uid 1000).
RUN useradd -m -u 1000 user

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    RAG_CONFIG=config.deploy.yaml \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Hand the whole app to the runtime user so it can read the baked index.
RUN chown -R user:user /app
USER user

# Build the index at image-build time (as 'user') so it is baked in and
# requires no writes or network at runtime.
RUN python run.py --config config.deploy.yaml pipeline || true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# If the baked index is somehow missing, rebuild; then serve.
CMD ["/bin/sh","-c","export RAG_CONFIG=config.deploy.yaml; [ -f data/processed/vectorstore/store.json ] || python run.py --config config.deploy.yaml pipeline; uvicorn rag.api.app:app --host 0.0.0.0 --port ${PORT}"]
