from __future__ import annotations

from src.case_schema import load_cases
from src.reviewers import make_reviewer


def test_uncertainty_aware_review_contains_claim_boundary() -> None:
    case = load_cases("cases/toy_cases.json")[0]
    review = make_reviewer("uncertainty_aware_review").review(case)
    text = " ".join(review["claim_boundary"]).lower()
    assert "not a real geotechnical safety prediction" in text
    assert review["recommended_review_status"] != "approve"


def test_no_v4_reviewer_returns_plain_approve() -> None:
    case = load_cases("cases/toy_cases.json")[0]
    for agent in [
        "generic_review",
        "surface_warning_review",
        "structural_memory_review",
        "relation_chain_review",
        "uncertainty_aware_review",
    ]:
        review = make_reviewer(agent).review(case)
        assert review["recommended_review_status"] != "approve"

