"""Chunking: split documents into overlapping chunks (recursive or sentence)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.ingest.clean import Document

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    # Find the best separator that keeps pieces under `size`.
    for sep in _SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            cur = ""
            for part in parts:
                candidate = part if not cur else cur + sep + part
                if len(candidate) <= size:
                    cur = candidate
                else:
                    if cur:
                        chunks.append(cur)
                    # part itself may exceed size -> recurse
                    if len(part) > size:
                        chunks.extend(_recursive_split(part, size, overlap))
                        cur = ""
                    else:
                        cur = part
            if cur:
                chunks.append(cur)
            return _apply_overlap(chunks, overlap)
    # No separators: hard slice.
    return _hard_slice(text, size, overlap)


def _hard_slice(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        out.append((tail + " " + cur).strip())
    return out


def _sentence_split(text: str, size: int, overlap: int) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    cur = ""
    for s in sents:
        candidate = s if not cur else cur + " " + s
        if len(candidate) <= size:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return _apply_overlap(chunks, overlap)


def chunk_document(
    doc: Document,
    strategy: str = "recursive",
    size: int = 600,
    overlap: int = 100,
    min_chars: int = 40,
) -> list[Chunk]:
    splitter = _sentence_split if strategy == "sentence" else _recursive_split
    pieces = splitter(doc.text, size, overlap)
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if len(piece) < min_chars:
            continue
        meta = dict(doc.metadata)
        meta["chunk_index"] = i
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{i}",
                doc_id=doc.doc_id,
                title=doc.title,
                text=piece,
                metadata=meta,
            )
        )
    return chunks


def chunk_documents(docs: list[Document], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, **kw))
    return out
