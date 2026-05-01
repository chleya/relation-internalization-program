from __future__ import annotations

from src.b4_intervention_metrics import B4_GATES, b4_intervention_score


def passing_metrics():
    return {
        "trace_guided_intervention_accuracy": 1.0,
        "intervention_region_accuracy": 1.0,
        "action_type_accuracy": 1.0,
        "outcome_improvement": 1.0,
        "intervention_vs_inspection_gain": 1.0,
        "wrong_region_penalty_sensitivity": 1.0,
        "family_specific_intervention_accuracy": 1.0,
        "recurrent_intervention_accuracy": 1.0,
        "field_intervention_accuracy": 1.0,
        "schema_intervention_accuracy": 1.0,
        "trace_ablation_intervention_drop": 1.0,
        "delay_ood_intervention_accuracy": 1.0,
        "gain_over_random": 1.0,
        "gain_over_saliency": 1.0,
        "gain_over_short_horizon": 1.0,
        "gain_over_inspect_only": 1.0,
        "oracle_intervention_score": 1.0,
        "random_intervention_score": 0.0,
    }


def test_any_core_gate_fail_forces_zero():
    metrics = passing_metrics()
    metrics["action_type_accuracy"] = 0.0
    assert b4_intervention_score(metrics, B4_GATES) == 0.0


def test_random_baseline_high_forces_zero():
    metrics = passing_metrics()
    metrics["random_intervention_score"] = 1.0
    assert b4_intervention_score(metrics, B4_GATES) == 0.0


def test_oracle_low_forces_zero():
    metrics = passing_metrics()
    metrics["oracle_intervention_score"] = 0.0
    assert b4_intervention_score(metrics, B4_GATES) == 0.0


def test_inspect_only_equal_to_intervention_forces_zero():
    metrics = passing_metrics()
    metrics["intervention_vs_inspection_gain"] = 0.0
    assert b4_intervention_score(metrics, B4_GATES) == 0.0


def test_all_gates_pass_returns_positive_score():
    assert b4_intervention_score(passing_metrics(), B4_GATES) > 0.0

