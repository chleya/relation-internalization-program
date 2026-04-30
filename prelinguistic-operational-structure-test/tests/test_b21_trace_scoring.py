from __future__ import annotations

from src.b21_trace_metrics import b21_trace_hardening_score


GATES = {
    "false_trace_rejection": 0.75,
    "trace_swap_sensitivity": 0.70,
    "trace_deletion_specificity_ratio": 1.50,
    "multi_source_conflict_resolution": 0.70,
    "noisy_trace_robustness": 0.70,
    "trace_length_extrapolation": 0.65,
    "trace_compression_survival": 0.65,
    "true_trace_intervention_drop": 0.20,
    "non_trace_stability": 0.70,
}


def passing_metrics():
    return {
        "false_trace_rejection": 0.90,
        "trace_swap_sensitivity": 0.80,
        "trace_deletion_specificity_ratio": 2.00,
        "multi_source_conflict_resolution": 0.80,
        "noisy_trace_robustness": 0.80,
        "trace_length_extrapolation": 0.70,
        "trace_compression_survival": 0.70,
        "true_trace_intervention_drop": 0.30,
        "non_trace_stability": 0.90,
    }


def test_any_core_gate_fail_forces_zero():
    metrics = passing_metrics()
    metrics["false_trace_rejection"] = 0.10
    assert b21_trace_hardening_score(metrics, GATES) == 0.0


def test_all_core_gates_pass_score_positive():
    assert b21_trace_hardening_score(passing_metrics(), GATES) > 0.0
