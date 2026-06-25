import numpy as np
import pytest

from rag.vectorstore.numpy_store import NumpyStore


def _data():
    ids = ["a", "b", "c"]
    vecs = np.eye(3, dtype=np.float32)
    texts = ["alpha", "beta", "gamma"]
    metas = [{"matrix": m} for m in ("A", "B", "C")]
    titles = ["TA", "TB", "TC"]
    return ids, vecs, texts, metas, titles


def test_numpy_store_roundtrip(tmp_path):
    s = NumpyStore(tmp_path / "vs")
    s.add(*_data())
    s.persist()
    hits = s.query(np.array([1, 0, 0], dtype=np.float32), top_k=2)
    assert hits[0].chunk_id == "a"
    assert hits[0].metadata["matrix"] == "A"

    s2 = NumpyStore(tmp_path / "vs")
    assert s2.load()
    assert s2.count() == 3
    assert s2.query(np.array([0, 1, 0], dtype=np.float32), top_k=1)[0].chunk_id == "b"


def test_faiss_store_roundtrip(tmp_path):
    pytest.importorskip("faiss")
    from rag.vectorstore.faiss_store import FaissStore
    s = FaissStore(tmp_path / "fs")
    s.add(*_data())
    s.persist()
    s2 = FaissStore(tmp_path / "fs")
    assert s2.load()
    assert s2.count() == 3
    assert s2.query(np.array([0, 0, 1], dtype=np.float32), top_k=1)[0].chunk_id == "c"
