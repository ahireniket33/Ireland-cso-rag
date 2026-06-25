import os

os.environ["RAG_CONFIG"] = "config.test.yaml"
os.environ.setdefault("RAG_LOG_LEVEL", "ERROR")


def _client(built_index):
    from fastapi.testclient import TestClient

    from rag.api.app import app
    return TestClient(app)


def test_health(built_index):
    c = _client(built_index)
    r = c.get("/health").json()
    assert r["status"] == "ok"
    assert r["indexed_chunks"] >= 3


def test_query_ok(built_index):
    c = _client(built_index)
    r = c.post("/query", json={"question": "What was the unemployment rate in Ireland in 2024?"}).json()
    assert not r["refused"]
    assert "4" in r["answer"]
    assert r["citations"][0]["matrix"] == "MUM01"


def test_query_off_domain(built_index):
    c = _client(built_index)
    r = c.post("/query", json={"question": "What is the capital of France?"}).json()
    assert r["refused"]


def test_validation_error(built_index):
    c = _client(built_index)
    assert c.post("/query", json={"question": ""}).status_code == 422
