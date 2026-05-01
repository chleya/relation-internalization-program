from __future__ import annotations

from src.b41_intervention_degeneracy_audit import B41_GATES, audit_action_value_leakage, b41_intervention_audit_score


def passing_metrics():
    return {
        "cross_model_exact_action_match_rate": 0.0,
        "exact_all_model_same_action_rate": 0.0,
        "fixed_action_type_rate": 0.0,
        "shared_action_policy_usage_rate": 0.0,
        "private_trace_action_score_usage_rate": 1.0,
        "fallback_usage_rate": 0.0,
        "mean_action_scorer_correlation": 0.0,
        "action_scorer_specificity": 1.0,
        "shared_action_policy_ablation_drop": 0.0,
        "private_action_retention_after_shared_ablation": 1.0,
        "wrong_action_penalty": 1.0,
        "wrong_region_penalty": 1.0,
        "wrong_action_wrong_region_penalty": 1.0,
        "random_intervention_score": 0.0,
        "oracle_intervention_score": 1.0,
        "trace_over_saliency_gain_margin": 1.0,
        "trace_over_short_horizon_gain_margin": 1.0,
        "trace_over_inspect_only_gain_margin": 1.0,
        "private_trace_ablation_drop": 1.0,
        "private_trace_over_non_trace_ratio": 10.0,
        "action_type_shift_after_trace_ablation": 1.0,
        "oracle_value_usage_rate": 0.0,
        "value_leakage_count": 0.0,
        "baseline_sanity_score": 1.0,
        "no_leakage_score": 1.0,
    }


def test_high_action_overlap_forces_zero():
    metrics = passing_metrics()
    metrics["cross_model_exact_action_match_rate"] = 1.0
    assert b41_intervention_audit_score(metrics, B41_GATES) == 0.0


def test_fixed_action_type_forces_zero():
    metrics = passing_metrics()
    metrics["fixed_action_type_rate"] = 1.0
    assert b41_intervention_audit_score(metrics, B41_GATES) == 0.0


def test_shared_policy_high_forces_zero():
    metrics = passing_metrics()
    metrics["shared_action_policy_usage_rate"] = 1.0
    assert b41_intervention_audit_score(metrics, B41_GATES) == 0.0


def test_oracle_value_leakage_forces_zero():
    metrics = passing_metrics()
    metrics["value_leakage_count"] = 1.0
    assert b41_intervention_audit_score(metrics, B41_GATES) == 0.0
    leakage = audit_action_value_leakage({"oracle_best_action": 1}, {}, {})
    assert leakage["value_leakage_count"] == 1


def test_random_high_and_oracle_low_force_zero():
    metrics = passing_metrics()
    metrics["random_intervention_score"] = 1.0
    assert b41_intervention_audit_score(metrics, B41_GATES) == 0.0
    metrics = passing_metrics()
    metrics["oracle_intervention_score"] = 0.0
    assert b41_intervention_audit_score(metrics, B41_GATES) == 0.0


def test_all_gates_pass_returns_positive_score():
    assert b41_intervention_audit_score(passing_metrics(), B41_GATES) > 0.0

