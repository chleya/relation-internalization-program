from __future__ import annotations

from src.b31_inspection_degeneracy_audit import b31_inspection_audit_score


GATES = {
    "max_cross_model_inspect_region_match_rate": 0.80,
    "max_shared_inspection_policy_usage_rate": 0.05,
    "min_private_inspection_score_usage_rate": 0.95,
    "fallback_usage_rate": 0.05,
    "min_inspection_scorer_specificity": 0.70,
    "max_inspection_scorer_correlation": 0.90,
    "max_shared_policy_ablation_drop": 0.05,
    "min_private_trace_ablation_drop": 0.20,
    "min_disagreement_inspection_divergence": 0.50,
    "min_family_aligned_inspection_rate": 0.60,
    "max_random_inspection_score": 0.25,
    "min_oracle_inspection_score": 0.95,
    "min_trace_over_saliency_gain_margin": 0.15,
    "min_trace_over_short_horizon_gain_margin": 0.15,
    "private_trace_over_non_trace_ratio": 1.50,
}


def passing_metrics():
    return {
        "cross_model_inspect_region_match_rate": 0.20,
        "shared_inspection_policy_usage_rate": 0.0,
        "private_trace_inspection_score_usage_rate": 1.0,
        "fallback_usage_rate": 0.0,
        "inspection_scorer_specificity": 0.80,
        "mean_inspection_scorer_correlation": 0.10,
        "shared_policy_ablation_drop": 0.0,
        "private_trace_ablation_drop": 0.90,
        "private_trace_over_non_trace_ratio": 10.0,
        "disagreement_inspection_divergence": 1.0,
        "family_aligned_inspection_rate": 1.0,
        "random_inspection_score": 0.05,
        "oracle_inspection_score": 1.0,
        "trace_over_saliency_gain_margin": 0.80,
        "trace_over_short_horizon_gain_margin": 0.80,
    }


def test_high_region_overlap_forces_zero():
    metrics = passing_metrics()
    metrics["cross_model_inspect_region_match_rate"] = 1.0
    assert b31_inspection_audit_score(metrics, GATES) == 0.0


def test_shared_policy_usage_forces_zero():
    metrics = passing_metrics()
    metrics["shared_inspection_policy_usage_rate"] = 0.5
    assert b31_inspection_audit_score(metrics, GATES) == 0.0


def test_private_trace_not_used_forces_zero():
    metrics = passing_metrics()
    metrics["private_trace_inspection_score_usage_rate"] = 0.0
    assert b31_inspection_audit_score(metrics, GATES) == 0.0


def test_random_baseline_high_forces_zero():
    metrics = passing_metrics()
    metrics["random_inspection_score"] = 0.8
    assert b31_inspection_audit_score(metrics, GATES) == 0.0


def test_oracle_baseline_low_forces_zero():
    metrics = passing_metrics()
    metrics["oracle_inspection_score"] = 0.5
    assert b31_inspection_audit_score(metrics, GATES) == 0.0


def test_trace_ablation_ratio_low_forces_zero():
    metrics = passing_metrics()
    metrics["private_trace_over_non_trace_ratio"] = 1.0
    assert b31_inspection_audit_score(metrics, GATES) == 0.0


def test_all_gates_pass_returns_positive_score():
    assert b31_inspection_audit_score(passing_metrics(), GATES) > 0.0
