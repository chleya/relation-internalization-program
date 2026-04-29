from __future__ import annotations

from src.case_schema import load_cases
from src.metrics_v4 import DEFAULT_GATES, aggregate_records, score_review
from src.reviewers import make_reviewer


def _score(agent: str) -> float:
    cases = load_cases("cases/toy_cases.json")
    reviewer = make_reviewer(agent)
    records = []
    for case in cases:
        records.append({"agent": agent, "case_id": case["case_id"], **score_review(case, reviewer.review(case))})
    return float(aggregate_records(records, DEFAULT_GATES)[0]["gated_v4_score"])


def test_negative_controls_have_zero_gated_score() -> None:
    for agent in ["generic_review", "surface_warning_review", "structural_memory_review"]:
        assert _score(agent) == 0.0

