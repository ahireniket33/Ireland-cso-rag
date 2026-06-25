from rag.ingest.chunk import chunk_document
from rag.ingest.clean import Document


def _doc(text):
    return Document(doc_id="d1", title="T", text=text, metadata={"matrix": "X"})


def test_short_text_single_chunk():
    chunks = chunk_document(_doc("A short but sufficiently long fact about Irish CSO statistics."), size=600, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1::0"


def test_long_text_splits_with_overlap():
    text = ". ".join(f"Sentence number {i} about Irish statistics" for i in range(60))
    chunks = chunk_document(_doc(text), strategy="recursive", size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 260 for c in chunks)  # size + overlap headroom
    assert all(c.metadata["matrix"] == "X" for c in chunks)


def test_min_chars_drop():
    chunks = chunk_document(_doc("tiny"), size=600, overlap=0, min_chars=40)
    assert chunks == []
