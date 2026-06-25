# PROGRESS LOG — Ireland CSO RAG

A running log of decisions, status, and rationale. Newest entries on top.

## Status summary
- [x] Approval Gate 1 — use case confirmed: **CSO economic & census data**
- [ ] Approval Gate 2 — deployment target (pending user confirmation)
- [ ] Approval Gate 3 — GitHub repo creation/push (needs user credentials)

## Decisions
1. **Use case:** Irish CSO economic & census statistics Q&A. Confirmed by user.
2. **Data source:** CSO PxStat open-data REST API (JSON-stat 2.0), CC BY 4.0.
   Verified reachable; returns rich aggregated time-series. Matrices: CPM01
   (CPI/inflation), MUM01 (unemployment), FY001A (census population).
3. **Table -> text strategy:** CSO data is tabular, which is weak for raw text
   RAG. We convert each observation into a natural-language "fact sentence"
   (e.g. "In 2023, the Consumer Price Index (All items) for Ireland was X")
   then chunk those. This makes statistics retrievable and citable.
4. **Embeddings:** swappable via config. Default `all-MiniLM-L6-v2`
   (sentence-transformers, free). A dependency-free **hashing** backend is
   included for hermetic/offline tests and CI (no HuggingFace download needed).
5. **Vector store:** Chroma (default, local-persistable) with a FAISS backend
   alternative, both behind a common interface.
6. **Generation:** swappable. Default **extractive** generator — needs no API
   key and incurs no cost; composes a grounded, cited answer from retrieved
   chunks. An OpenAI-compatible LLM backend is included and config-selectable.
7. **Guardrails:** input validation + off-domain/prompt-injection blocking,
   retrieval-confidence threshold, mandatory inline citations, output
   groundedness/faithfulness check, PII filtering on input & output.

## Environment notes
- Dev sandbox can reach PyPI/GitHub but NOT cso.ie/huggingface. Therefore:
  live download script targets CSO API (runs on user machine / deploy / CI);
  bundled real sample + hashing embedder make the suite run fully offline here.

## Log
- 2026-06-25: Gate 1 approved. Scaffolded repo, config, licenses, progress log.
- 2026-06-25: Built full pipeline (ingest/embed/store/retrieve/generate),
  guardrails, FastAPI, single entrypoint. 30 pytest tests PASS, ruff lint CLEAN.
  Eval suite PASS: retrieval 1.0, answer acc 1.0, faithfulness 1.0, refusal 1.0.
- 2026-06-25: Added Dockerfile, docker-compose, GitHub Actions CI (lint+test+
  build), Hugging Face Spaces deploy scaffold, comprehensive README.
- 2026-06-25: Local git repo initialised with incremental commits. AWAITING
  Gate 2 (deployment target) and Gate 3 (GitHub push) approvals.

