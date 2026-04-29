from __future__ import annotations

from src.case_schema import load_cases
from src.metrics_v4 import DEFAULT_GATES, aggregate_records, score_review
from src.reviewers import make_reviewer


def _summary_for(agent: str) -> dict[str, object]:
    cases = load_cases("cases/toy_cases.json")
    reviewer = make_reviewer(agent)
    records = []
    for case in cases:
        review = reviewer.review(case)
        records.append({"agent": agent, "case_id": case["case_id"], **score_review(case, review)})
    return aggregate_records(records, DEFAULT_GATES)[0]


def test_uncertainty_aware_review_passes_v4_gate() -> None:
    summary = _summary_for("uncertainty_aware_review")
    assert summary["gated_v4_score"] > 0.9
    assert summary["relation_chain_specificity"] >= 0.8
    assert summary["uncertainty_takeover_quality"] >= 0.8


def test_relation_chain_review_fails_uncertainty_gate() -> None:
    summary = _summary_for("relation_chain_review")
    assert summary["relation_chain_specificity"] >= 0.8
    assert summary["action_point_mapping"] >= 0.8
    assert summary["uncertainty_takeover_quality"] < 0.8
    assert summary["gated_v4_score"] == 0.0

