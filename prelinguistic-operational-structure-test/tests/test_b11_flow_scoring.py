from __future__ import annotations

from src.b11_flow_attacks import compute_b11_hardening_score


GATES = {
    "dynamic_decoy_rejection": 0.80,
    "delayed_checkpoint_accuracy": 0.75,
    "competing_checkpoint_choice": 0.75,
    "relocation_ood_stability": 0.75,
    "causal_over_visual_deletion_ratio": 1.50,
    "anti_prior_survival": 0.70,
    "causal_endpoint_shift": 0.25,
}


def passing_metrics():
    return {
        "candidate_gate_preserved": 1.0,
        "dynamic_decoy_rejection": 0.90,
        "delayed_checkpoint_accuracy": 0.80,
        "competing_checkpoint_choice": 0.80,
        "relocation_ood_stability": 0.80,
        "causal_over_visual_deletion_ratio": 1.70,
        "anti_prior_survival": 0.75,
        "causal_endpoint_shift": 0.30,
    }


def test_if_any_gate_fails_score_is_zero():
    metrics = passing_metrics()
    metrics["competing_checkpoint_choice"] = 0.20
    assert compute_b11_hardening_score(metrics, GATES) == 0.0


def test_if_all_gates_pass_score_is_positive():
    assert compute_b11_hardening_score(passing_metrics(), GATES) > 0.0


def test_candidate_gate_preserved_failure_forces_zero():
    metrics = passing_metrics()
    metrics["candidate_gate_preserved"] = 0.0
    assert compute_b11_hardening_score(metrics, GATES) == 0.0
