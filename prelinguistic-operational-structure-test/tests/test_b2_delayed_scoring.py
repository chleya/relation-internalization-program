from __future__ import annotations

from src.b2_delayed_metrics import b2_delayed_score


GATES = {
    "delayed_checkpoint_accuracy": 0.75,
    "multi_delay_stability": 0.70,
    "early_saliency_rejection": 0.75,
    "delay_ood_generalization": 0.70,
    "causal_trace_intervention_drop": 0.20,
    "non_trace_stability": 0.70,
    "delayed_endpoint_shift": 0.25,
}


def passing_metrics():
    return {
        "delayed_checkpoint_accuracy": 0.90,
        "multi_delay_stability": 0.80,
        "early_saliency_rejection": 0.90,
        "delay_ood_generalization": 0.80,
        "causal_trace_intervention_drop": 0.30,
        "non_trace_stability": 0.90,
        "delayed_endpoint_shift": 0.30,
    }


def test_any_failed_core_gate_forces_zero():
    metrics = passing_metrics()
    metrics["delay_ood_generalization"] = 0.20
    assert b2_delayed_score(metrics, GATES) == 0.0


def test_all_gates_pass_score_positive():
    assert b2_delayed_score(passing_metrics(), GATES) > 0.0
