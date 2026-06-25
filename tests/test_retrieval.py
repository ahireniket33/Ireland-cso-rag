from rag.embeddings.factory import build_embedder
from rag.retrieval.retriever import Retriever
from rag.vectorstore.factory import build_store


def test_threshold_filters(built_index):
    cfg = built_index
    emb = build_embedder(cfg)
    store = build_store(cfg)
    store.load()
    r = Retriever(cfg, emb, store)
    hits = r.retrieve("What was the population of Ireland in 2022?")
    assert hits
    assert all(h.score >= r.threshold for h in hits)
    assert hits[0].metadata["matrix"] == "FY001A"
