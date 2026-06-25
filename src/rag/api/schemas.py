"""API request/response schemas (typed, validated)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="A question about Irish CSO statistics.")


class CitationModel(BaseModel):
    matrix: str
    title: str
    url: str
    source: str = ""
    score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    reason: str
    citations: list[CitationModel] = []
    faithfulness: float
    num_retrieved: int
    flags: list[str] = []
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    embeddings_backend: str
    vectorstore_backend: str
    generation_backend: str
    version: str
