from __future__ import annotations

from src.b3_active_inspection_metrics import b3_active_inspection_score


GATES = {
    "trace_guided_inspection_accuracy": 0.75,
    "trace_vs_saliency_rejection": 0.75,
    "delayed_information_gain": 0.20,
    "inspection_value_gain_over_random": 0.20,
    "inspection_value_gain_over_saliency": 0.15,
    "inspection_value_gain_over_short_horizon": 0.15,
    "trace_ablation_inspection_drop": 0.20,
    "delay_ood_inspection_accuracy": 0.65,
    "oracle_inspection_score": 0.95,
    "random_inspection_score": 0.25,
}


def passing_metrics():
    return {
        "trace_guided_inspection_accuracy": 0.90,
        "trace_vs_saliency_rejection": 0.90,
        "delayed_information_gain": 0.80,
        "inspection_value_gain_over_random": 0.70,
        "inspection_value_gain_over_saliency": 0.60,
        "inspection_value_gain_over_short_horizon": 0.60,
        "trace_ablation_inspection_drop": 0.50,
        "delay_ood_inspection_accuracy": 0.80,
        "oracle_inspection_score": 1.0,
        "random_inspection_score": 0.10,
    }


def test_any_core_gate_fail_forces_zero():
    metrics = passing_metrics()
    metrics["trace_guided_inspection_accuracy"] = 0.0
    assert b3_active_inspection_score(metrics, GATES) == 0.0


def test_random_baseline_high_forces_zero():
    metrics = passing_metrics()
    metrics["random_inspection_score"] = 0.80
    assert b3_active_inspection_score(metrics, GATES) == 0.0


def test_oracle_baseline_low_forces_zero():
    metrics = passing_metrics()
    metrics["oracle_inspection_score"] = 0.50
    assert b3_active_inspection_score(metrics, GATES) == 0.0


def test_all_gates_pass_returns_positive_score():
    assert b3_active_inspection_score(passing_metrics(), GATES) > 0.0
