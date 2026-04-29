from __future__ import annotations

from src.case_schema import load_cases, validate_case


def test_governance_cases_load_and_disallow_auto_approval() -> None:
    cases = load_cases("cases/governance_cases.json")
    assert len(cases) >= 3
    for case in cases:
        validate_case(case)
        assert case["expected_auto_approval"] is False
        assert case["review_status"] != "approve"

