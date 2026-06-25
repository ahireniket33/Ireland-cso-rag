from rag.ingest.clean import deduplicate, observations_to_documents
from rag.ingest.jsonstat import parse_dataset


def test_documents_have_citation(sample_doc):
    meta, obs = parse_dataset(sample_doc)
    docs = observations_to_documents(meta, obs, min_year=2015)
    assert docs
    d = docs[0]
    assert d.metadata["matrix"] == "CPM01"
    assert d.metadata["license"] == "CC BY 4.0"
    assert "7.8" in d.text
    assert "inflation" in d.text.lower()  # alias enrichment


def test_dedup():
    from rag.ingest.clean import Document
    a = Document("1", "t", "same text here", {})
    b = Document("2", "t", "same text here", {})
    assert len(deduplicate([a, b])) == 1
