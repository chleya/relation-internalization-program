from __future__ import annotations

from src.b42_action_type_metrics import B42_GATES, b42_action_type_score


def passing_metrics():
    return {
        "fixed_action_type_rate": 0.5,
        "action_type_accuracy": 1.0,
        "region_accuracy": 1.0,
        "joint_region_action_accuracy": 1.0,
        "correct_region_wrong_action_penalty": 1.0,
        "action_type_counterfactual_sensitivity": 1.0,
        "family_action_diversity": 0.5,
        "family_action_mapping_accuracy": 1.0,
        "action_type_shift_after_trace_ablation": 1.0,
        "action_type_ablation_drop": 1.0,
        "region_stability_after_action_ablation": 1.0,
        "gain_over_fixed_action_baseline": 1.0,
        "gain_over_random_action_type": 1.0,
        "gain_over_saliency": 1.0,
        "gain_over_short_horizon": 1.0,
        "action_type_ood_accuracy": 1.0,
        "oracle_action_type_score": 1.0,
        "value_leakage_count": 0.0,
    }


def test_fixed_action_type_high_forces_zero():
    metrics = passing_metrics()
    metrics["fixed_action_type_rate"] = 1.0
    assert b42_action_type_score(metrics, B42_GATES) == 0.0


def test_low_penalty_forces_zero():
    metrics = passing_metrics()
    metrics["correct_region_wrong_action_penalty"] = 0.0
    assert b42_action_type_score(metrics, B42_GATES) == 0.0


def test_low_fixed_baseline_gain_forces_zero():
    metrics = passing_metrics()
    metrics["gain_over_fixed_action_baseline"] = 0.0
    assert b42_action_type_score(metrics, B42_GATES) == 0.0


def test_low_oracle_forces_zero():
    metrics = passing_metrics()
    metrics["oracle_action_type_score"] = 0.0
    assert b42_action_type_score(metrics, B42_GATES) == 0.0


def test_all_gates_pass_returns_positive_score():
    assert b42_action_type_score(passing_metrics(), B42_GATES) > 0.0

