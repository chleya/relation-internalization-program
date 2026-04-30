from __future__ import annotations

from src.scoring import plos_candidate_score

from .conftest import small_config


def _passing_metrics():
    metrics = {key: 1.0 for key in small_config()["gates"]}
    metrics["slot_causal_drop"] = 0.3
    metrics["field_causal_drop"] = 0.3
    return metrics


def test_scoring_returns_zero_when_behavior_gate_fails() -> None:
    metrics = _passing_metrics()
    metrics["identity_after_occlusion"] = 0.0
    assert plos_candidate_score(metrics, small_config()["gates"]) == 0.0


def test_scoring_returns_zero_when_all_structure_gates_fail() -> None:
    metrics = _passing_metrics()
    for key in [
        "slot_causal_drop",
        "event_latent_causal_drop",
        "relation_edge_causal_drop",
        "inspection_map_causal_drop",
        "field_causal_drop",
        "critical_field_locality",
        "noncritical_field_invariance",
    ]:
        metrics[key] = 0.0
    assert plos_candidate_score(metrics, small_config()["gates"]) == 0.0


def test_scoring_returns_zero_when_ood_gate_fails() -> None:
    metrics = _passing_metrics()
    metrics["ood_trajectory_generalization"] = 0.0
    assert plos_candidate_score(metrics, small_config()["gates"]) == 0.0
