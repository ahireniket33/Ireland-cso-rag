"""Run the gold Q&A set through the pipeline and score retrieval + faithfulness."""
from __future__ import annotations

from pathlib import Path

import yaml

from rag.config import Config
from rag.logging_utils import get_logger
from rag.pipeline import RAGPipeline

log = get_logger("rag.eval")


def load_gold(path: Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_eval(cfg: Config, pipeline: RAGPipeline | None = None) -> dict:
    gold_path = (cfg.root / cfg.get("eval", "gold_file")).resolve()
    gold = load_gold(gold_path)
    pipe = pipeline or RAGPipeline(cfg)

    answerable = [g for g in gold if not g.get("should_refuse")]
    refusable = [g for g in gold if g.get("should_refuse")]

    retrieval_hits = 0
    substring_hits = 0
    faithfulness_sum = 0.0
    refusal_correct = 0
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_rank_sum = 0.0
    details = []

    for g in answerable:
        r = pipe.answer(g["question"])
        cited = {c.matrix for c in r.citations}
        rel = g["expected_matrix"] in cited
        sub = (not r.refused) and (g.get("expect_substring", "") in r.answer)

        # Rank of the first retrieved chunk from the expected dataset (Hit@k/MRR).
        ranked = pipe.retriever.retrieve(g["question"])
        rank = next((i + 1 for i, h in enumerate(ranked)
                     if h.metadata.get("matrix") == g["expected_matrix"]), None)
        hit_at_1 += int(rank == 1)
        hit_at_3 += int(rank is not None and rank <= 3)
        reciprocal_rank_sum += (1.0 / rank) if rank else 0.0

        retrieval_hits += int(rel)
        substring_hits += int(sub)
        faithfulness_sum += r.faithfulness
        details.append({
            "q": g["question"], "refused": r.refused, "cited": sorted(cited),
            "expected": g["expected_matrix"], "retrieval_ok": rel,
            "rank": rank, "substring_ok": sub, "faithfulness": r.faithfulness,
        })

    for g in refusable:
        r = pipe.answer(g["question"])
        ok = r.refused
        refusal_correct += int(ok)
        details.append({"q": g["question"], "refused": r.refused,
                        "should_refuse": True, "refusal_ok": ok})

    n_ans = max(1, len(answerable))
    n_ref = max(1, len(refusable))
    retrieval_relevance = retrieval_hits / n_ans
    answer_accuracy = substring_hits / n_ans
    faithfulness = faithfulness_sum / n_ans
    refusal_rate = refusal_correct / n_ref
    hit1 = hit_at_1 / n_ans
    hit3 = hit_at_3 / n_ans
    mrr = reciprocal_rank_sum / n_ans

    rel_min = cfg.get("eval", "retrieval_relevance_min", default=0.6)
    faith_min = cfg.get("eval", "faithfulness_min", default=0.6)
    passed = (retrieval_relevance >= rel_min
              and faithfulness >= faith_min
              and refusal_rate >= 0.99
              and answer_accuracy >= rel_min)

    report = {
        "passed": passed,
        "metrics": {
            "retrieval_relevance": round(retrieval_relevance, 3),
            "hit_at_1": round(hit1, 3),
            "hit_at_3": round(hit3, 3),
            "mrr": round(mrr, 3),
            "answer_accuracy": round(answer_accuracy, 3),
            "faithfulness": round(faithfulness, 3),
            "refusal_accuracy": round(refusal_rate, 3),
        },
        "thresholds": {"retrieval_relevance_min": rel_min, "faithfulness_min": faith_min},
        "counts": {"answerable": len(answerable), "refusable": len(refusable)},
        "details": details,
    }
    log.info("EVAL: %s", report["metrics"])
    return report
