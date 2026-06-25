from rag.eval.evaluate import run_eval


def test_eval_passes(pipeline, built_index):
    report = run_eval(built_index, pipeline=pipeline)
    assert report["passed"], report["metrics"]
    m = report["metrics"]
    assert m["retrieval_relevance"] >= 0.8
    assert m["faithfulness"] >= 0.6
    assert m["refusal_accuracy"] == 1.0
