from __future__ import annotations

from src.b21a_audit_metrics import b21a_degeneracy_audit_score


GATES = {
    "max_random_baseline_score": 0.20,
    "min_oracle_score": 0.95,
    "max_leakage_score": 0.00,
    "min_intervention_applicability_rate": 0.95,
    "min_trace_family_ablation_drop": 0.20,
    "min_no_trace_ablation_drop": 0.30,
}


def passing_metrics():
    return {
        "leakage_count": 0.0,
        "intervention_applicability_rate": 1.0,
        "random_b21_score": 0.0,
        "oracle_b21_score": 1.0,
        "trace_family_ablation_drop": 0.8,
        "no_trace_ablation_drop": 0.8,
    }


def test_leakage_count_forces_zero():
    metrics = passing_metrics()
    metrics["leakage_count"] = 1.0
    assert b21a_degeneracy_audit_score(metrics, GATES) == 0.0


def test_random_baseline_high_forces_zero():
    metrics = passing_metrics()
    metrics["random_b21_score"] = 0.5
    assert b21a_degeneracy_audit_score(metrics, GATES) == 0.0


def test_oracle_baseline_low_forces_zero():
    metrics = passing_metrics()
    metrics["oracle_b21_score"] = 0.5
    assert b21a_degeneracy_audit_score(metrics, GATES) == 0.0


def test_no_ablation_drop_forces_zero():
    metrics = passing_metrics()
    metrics["trace_family_ablation_drop"] = 0.0
    assert b21a_degeneracy_audit_score(metrics, GATES) == 0.0


def test_unexplained_exact_prediction_degeneracy_forces_zero():
    metrics = passing_metrics()
    metrics["cross_model_exact_prediction_match_rate"] = 1.0
    metrics["all_attacks_identical_flag"] = 1.0
    metrics["gt_region_match_rate"] = 0.70
    assert b21a_degeneracy_audit_score(metrics, GATES) == 0.0


def test_all_gates_pass_score_positive():
    assert b21a_degeneracy_audit_score(passing_metrics(), GATES) > 0.0
