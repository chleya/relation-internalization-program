from __future__ import annotations

from src.analyze_results import case_variant, inspect_error, merged_rows


def test_case_variant_extracts_budgeted_stress_variant() -> None:
    assert case_variant("seed2_budgeted_inspect_many_distractors", "budgeted_inspect") == "many_distractors"


def test_inspect_error_classifies_no_inspect() -> None:
    expected = {"inspect": "q2"}
    response = {"inspect": "none"}
    assert inspect_error(expected, response, 0.0) == "no_inspect"


def test_inspect_error_classifies_over_inspect() -> None:
    expected = {"inspect": "none"}
    response = {"inspect": "q2"}
    assert inspect_error(expected, response, 0.0) == "over_inspect"


def test_merged_rows_joins_record_and_raw_payload() -> None:
    records = [
        {
            "solver": "llama_cpp",
            "seed": "0",
            "case_id": "seed0_budgeted_inspect_noise_first",
            "gate": "budgeted_inspect",
            "case_score": "0.0",
            "answer_match": "1.0",
            "inspect_match": "0.0",
            "uncertain_match": "1.0",
            "audit_links_match": "0.0",
            "query_sets_match": "1.0",
            "error": "",
        }
    ]
    raw = {
        ("llama_cpp", "0", "seed0_budgeted_inspect_noise_first"): {
            "expected": {"inspect": "q2", "uncertain": True, "audit_links": ["q2 -> z9"]},
            "response": {"inspect": "none", "uncertain": True, "audit_links": []},
        }
    }
    rows = merged_rows(records, raw)
    assert rows[0]["variant"] == "noise_first"
    assert rows[0]["expected_inspect"] == "q2"
    assert rows[0]["actual_inspect"] == "none"
    assert rows[0]["inspect_error"] == "no_inspect"
