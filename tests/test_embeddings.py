import numpy as np

from rag.embeddings.hashing import HashingEmbedder


def test_shape_and_norm():
    e = HashingEmbedder(dim=128)
    v = e.embed(["Irish inflation 2022", "population of Ireland"])
    assert v.shape == (2, 128)
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_deterministic():
    e = HashingEmbedder(dim=64)
    assert np.array_equal(e.embed(["hello world"]), e.embed(["hello world"]))


def test_similar_texts_closer():
    e = HashingEmbedder(dim=256)
    a, b, c = e.embed(["inflation rate ireland", "ireland inflation rate cpi", "pizza topping"])
    assert float(a @ b) > float(a @ c)
