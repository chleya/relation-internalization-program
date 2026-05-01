from __future__ import annotations

from src.b32_mechanism_metrics import B32_GATES, b32_mechanism_inspection_score


def passing_metrics():
    return {
        "family_specific_inspection_accuracy": 1.0,
        "recurrent_goal_accuracy": 1.0,
        "field_goal_accuracy": 1.0,
        "schema_goal_accuracy": 1.0,
        "mechanism_disagreement_rate": 1.0,
        "cross_model_same_region_rate": 0.0,
        "task_conditioned_switch_accuracy": 1.0,
        "inspect_value_decomposition_alignment": 1.0,
        "family_specific_trace_ablation_drop": 1.0,
        "non_target_family_stability": 1.0,
        "gain_over_random": 1.0,
        "gain_over_saliency": 1.0,
        "gain_over_short_horizon": 1.0,
        "oracle_family_inspection_score": 1.0,
    }


def test_low_family_accuracy_forces_zero():
    metrics = passing_metrics()
    metrics["family_specific_inspection_accuracy"] = 0.0
    assert b32_mechanism_inspection_score(metrics, B32_GATES) == 0.0


def test_low_switch_accuracy_forces_zero():
    metrics = passing_metrics()
    metrics["task_conditioned_switch_accuracy"] = 0.0
    assert b32_mechanism_inspection_score(metrics, B32_GATES) == 0.0


def test_low_disagreement_forces_zero():
    metrics = passing_metrics()
    metrics["mechanism_disagreement_rate"] = 0.0
    assert b32_mechanism_inspection_score(metrics, B32_GATES) == 0.0


def test_high_same_region_rate_forces_zero():
    metrics = passing_metrics()
    metrics["cross_model_same_region_rate"] = 1.0
    assert b32_mechanism_inspection_score(metrics, B32_GATES) == 0.0


def test_all_gates_pass_returns_positive_score():
    assert b32_mechanism_inspection_score(passing_metrics(), B32_GATES) > 0.0
