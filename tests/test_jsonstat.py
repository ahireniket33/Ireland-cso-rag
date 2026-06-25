from rag.ingest.jsonstat import parse_dataset


def test_parse_sample(sample_doc):
    meta, obs = parse_dataset(sample_doc)
    assert meta.matrix == "CPM01"
    assert "Central Statistics Office" in meta.source
    assert len(obs) >= 5
    years = {o.year for o in obs}
    assert 2022 in years
    o2022 = next(o for o in obs if o.year == 2022)
    assert o2022.value == 7.8
    assert o2022.unit == "%"


def test_rejects_non_dataset():
    import pytest
    with pytest.raises(ValueError):
        parse_dataset({"class": "collection"})
