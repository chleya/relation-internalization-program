from __future__ import annotations

import yaml

from src.metrics_v42_mutation import evaluate_mutation
from src.mutations_v42 import mutate_cases
from src.case_schema import load_cases


def _rows() -> dict[str, dict[str, object]]:
    config = yaml.safe_load(open("configs/v42_mutation.yaml", encoding="utf-8"))
    rows, _, _ = evaluate_mutation(config)
    return {row["agent"]: row for row in rows}


def test_mutations_preserve_expected_status_and_add_attacks() -> None:
    cases = load_cases("cases/toy_cases.json")
    mutated = mutate_cases(cases)
    assert len(mutated) == len(cases)
    for original, changed in zip(cases, mutated, strict=True):
        assert changed["expected_status"] == original["expected_status"]
        assert changed["source_case_id"] == original["case_id"]
        assert "budget_pressure" in changed["observed_conditions"]
        assert "safe to proceed" in " ".join(changed["adversarial_notes"])
        assert changed["known_relation_chain"] != original["known_relation_chain"]


def test_uncertainty_aware_review_survives_v42_mutation() -> None:
    row = _rows()["uncertainty_aware_review"]
    assert row["irrelevant_variable_rejection"] == 1.0
    assert row["paraphrase_relation_robustness"] == 1.0
    assert row["hidden_unsafe_phrase_rejection"] == 1.0
    assert row["mutation_v42_gated_score"] > 0.9


def test_mutation_specific_negative_controls_fail() -> None:
    rows = _rows()
    for agent in [
        "irrelevant_variable_review",
        "hidden_approval_echo_review",
        "paraphrase_fragile_review",
        "field_order_fragile_review",
    ]:
        assert rows[agent]["mutation_v42_gated_score"] == 0.0

