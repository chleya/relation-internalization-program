from __future__ import annotations

from src.case_schema import load_cases


def test_audit_cases_load_with_multiple_reviews() -> None:
    cases = load_cases("cases/audit_cases.json")
    assert len(cases) >= 3
    assert all(len(case["reviews"]) >= 2 for case in cases)
    assert all(case["expected_material_disagreement"] is True for case in cases)

