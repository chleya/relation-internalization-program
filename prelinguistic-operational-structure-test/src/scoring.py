from __future__ import annotations

from statistics import mean
from typing import Any


BEHAVIOR_KEYS = [
    "identity_after_occlusion",
    "identity_after_crossing",
    "event_boundary_alignment",
    "intervention_sensitivity",
    "relation_locality",
    "noncausal_region_invariance",
    "critical_region_selection_accuracy",
    "inspection_value_gain",
]

STRUCTURE_KEYS = [
    "slot_causal_drop",
    "slot_swap_consistency",
    "event_latent_causal_drop",
    "relation_edge_causal_drop",
    "inspection_map_causal_drop",
    "field_causal_drop",
    "critical_field_locality",
    "noncritical_field_invariance",
]

OOD_KEYS = ["ood_trajectory_generalization"]


def behavior_score(metrics: dict[str, float]) -> float:
    return _average(metrics, BEHAVIOR_KEYS)


def structure_intervention_score(metrics: dict[str, float]) -> float:
    return _average(metrics, STRUCTURE_KEYS)


def ood_score(metrics: dict[str, float]) -> float:
    return _average(metrics, OOD_KEYS)


def plos_candidate_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    behavior_gate_keys = [
        "identity_after_occlusion",
        "identity_after_crossing",
        "event_boundary_alignment",
        "intervention_sensitivity",
        "relation_locality",
        "noncausal_region_invariance",
        "critical_region_selection_accuracy",
        "ood_trajectory_generalization",
    ]
    if any(metrics.get(key, 0.0) < gates.get(key, 0.0) for key in behavior_gate_keys):
        return 0.0

    relevant_structure = [
        "slot_causal_drop",
        "event_latent_causal_drop",
        "relation_edge_causal_drop",
        "inspection_map_causal_drop",
        "field_causal_drop",
        "critical_field_locality",
        "noncritical_field_invariance",
    ]
    if not any(metrics.get(key, 0.0) >= gates.get(key, 1.0) for key in relevant_structure):
        return 0.0

    if any(metrics.get(key, 0.0) < gates.get(key, 0.0) for key in OOD_KEYS):
        return 0.0

    return 0.35 * behavior_score(metrics) + 0.40 * structure_intervention_score(metrics) + 0.25 * ood_score(metrics)


def _average(metrics: dict[str, float], keys: list[str]) -> float:
    values = [float(metrics.get(key, 0.0)) for key in keys if key in metrics]
    return mean(values) if values else 0.0


def merge_scores(behavior: dict[str, float], structure: dict[str, float], ood: dict[str, float], gates: dict[str, float]) -> dict[str, float]:
    metrics = {**behavior, **structure, **ood}
    metrics["behavior_score"] = behavior_score(metrics)
    metrics["structure_intervention_score"] = structure_intervention_score(metrics)
    metrics["ood_score"] = ood_score(metrics)
    metrics["plos_candidate_score"] = plos_candidate_score(metrics, gates)
    return metrics
