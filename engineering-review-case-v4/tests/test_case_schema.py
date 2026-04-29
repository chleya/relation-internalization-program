from __future__ import annotations

from src.case_schema import load_cases, validate_case


def test_cases_load_and_have_no_plain_approve_status() -> None:
    cases = load_cases("cases/toy_cases.json")
    assert len(cases) >= 5
    for case in cases:
        validate_case(case)
        assert case["expected_status"] != "approve"


def test_cases_contain_relation_chains_and_uncertainties() -> None:
    cases = load_cases("cases/toy_cases.json")
    assert all(case["known_relation_chain"] for case in cases)
    assert all("->" in case["known_relation_chain"][0] for case in cases)
    assert any(case["expected_status"] == "takeover_required" for case in cases)

