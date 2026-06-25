# Deployment Guide

## Recommended: Hugging Face Spaces (Docker SDK) — free CPU tier

**Why HF Spaces:** free persistent CPU tier with enough RAM for
`sentence-transformers`, native access to the Hugging Face model hub (no model
download throttling), Docker support so we ship the exact same image as locally,
and a public HTTPS URL out of the box. Render's free tier (512 MB) risks OOM
when loading the embedding model and spins down on idle; Fly.io and Railway now
require a card / are trial-only. HF Spaces is the most reliable zero-cost option
for this stack.

### Steps (you run these — they need your Hugging Face account)
1. Create a new Space: https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: **CPU basic (free)**
2. Push this repository to the Space's git remote. The Space's `README.md` must
   contain the front-matter in `deploy/huggingface/README.md` (copy it to the
   repo root `README.md` *for the Space*, or merge the front-matter block in).
3. The Space builds the root `Dockerfile` and starts the API on port 8000
   (declared via `app_port: 8000`).
4. Smoke test (replace with your Space URL):
   ```bash
   curl https://<user>-ireland-cso-rag.hf.space/health
   curl -X POST https://<user>-ireland-cso-rag.hf.space/query \
        -H 'Content-Type: application/json' \
        -d '{"question":"What was the rate of inflation in Ireland in 2022?"}'
   ```

## Alternative: Render (free web service)
- New > Web Service > Build from Dockerfile.
- Set `PORT=8000` (Render injects `PORT`; the image already honours it).
- Note: free instances sleep on idle and have 512 MB RAM. To stay within it,
  set `embeddings.backend: hashing` and `vectorstore.backend: numpy` in
  `config.yaml` (no model download, tiny memory) — lower retrieval quality but
  fully self-contained.

## Local (Docker)
```bash
docker compose up --build
# then:
curl http://localhost:8000/health
```
