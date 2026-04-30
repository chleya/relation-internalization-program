from __future__ import annotations

from typing import Any

import numpy as np


def prediction_delta(base: np.ndarray, changed: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(base) - np.asarray(changed))))


def slot_causal_drop(result: dict[str, Any]) -> float:
    return _drop_if_applicable(result, "slot")


def slot_swap_consistency(result: dict[str, Any]) -> float:
    if not result.get("applicable"):
        return 0.0
    return float(np.clip(prediction_delta(result["base_future_frames"], result["future_frames"]) * 5.0, 0.0, 1.0))


def non_target_stability(result: dict[str, Any]) -> float:
    if not result.get("applicable"):
        return 0.0
    delta = prediction_delta(result["base_future_frames"], result["future_frames"])
    return float(np.clip(1.0 - delta * 2.0, 0.0, 1.0))


def event_latent_causal_drop(result: dict[str, Any]) -> float:
    return _drop_if_applicable(result, "event_latent_perturbation")


def event_shift_effect(result: dict[str, Any]) -> float:
    return _generic_drop(result)


def non_event_stability(result: dict[str, Any]) -> float:
    return non_target_stability(result)


def relation_edge_causal_drop(result: dict[str, Any]) -> float:
    return _drop_if_applicable(result, "relation_edge_ablation")


def relation_locality_under_ablation(result: dict[str, Any]) -> float:
    return non_target_stability(result)


def noncausal_edge_invariance(result: dict[str, Any]) -> float:
    return non_target_stability(result)


def inspection_map_causal_drop(result: dict[str, Any]) -> float:
    return _drop_if_applicable(result, "inspection")


def inspection_value_loss_after_perturbation(result: dict[str, Any]) -> float:
    return _generic_drop(result)


def prediction_stability_under_inspection_perturbation(result: dict[str, Any]) -> float:
    return non_target_stability(result)


def field_causal_drop(result: dict[str, Any]) -> float:
    return _drop_if_applicable(result, "field")


def critical_field_locality(result: dict[str, Any]) -> float:
    return non_target_stability(result)


def noncritical_field_invariance(result: dict[str, Any]) -> float:
    return non_target_stability(result)


def intervention_metrics(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        "slot_causal_drop": max(slot_causal_drop(results.get("slot_removal", {})), slot_swap_consistency(results.get("slot_swap", {}))),
        "slot_swap_consistency": slot_swap_consistency(results.get("slot_swap", {})),
        "non_target_stability": max(non_target_stability(result) for result in results.values()) if results else 0.0,
        "event_latent_causal_drop": event_latent_causal_drop(results.get("event_latent_perturbation", {})),
        "event_shift_effect": event_shift_effect(results.get("event_latent_perturbation", {})),
        "non_event_stability": non_event_stability(results.get("event_latent_perturbation", {})),
        "relation_edge_causal_drop": relation_edge_causal_drop(results.get("relation_edge_ablation", {})),
        "relation_locality_under_ablation": relation_locality_under_ablation(results.get("relation_edge_ablation", {})),
        "noncausal_edge_invariance": noncausal_edge_invariance(results.get("relation_edge_ablation", {})),
        "inspection_map_causal_drop": max(
            inspection_map_causal_drop(results.get("inspection_map_shuffle", {})),
            inspection_value_loss_after_perturbation(results.get("inspection_topk_zero", {})),
        ),
        "inspection_value_loss_after_perturbation": inspection_value_loss_after_perturbation(results.get("inspection_topk_zero", {})),
        "prediction_stability_under_inspection_perturbation": prediction_stability_under_inspection_perturbation(results.get("inspection_map_shuffle", {})),
        "field_causal_drop": max(field_causal_drop(results.get("field_patch_mask", {})), field_causal_drop(results.get("critical_field_zero", {}))),
        "critical_field_locality": critical_field_locality(results.get("critical_field_zero", {})),
        "noncritical_field_invariance": noncritical_field_invariance(results.get("field_patch_mask", {})),
    }


def _generic_drop(result: dict[str, Any]) -> float:
    if not result.get("applicable"):
        return 0.0
    return float(np.clip(prediction_delta(result["base_future_frames"], result["future_frames"]) * 5.0, 0.0, 1.0))


def _drop_if_applicable(result: dict[str, Any], target: str) -> float:
    if not result.get("applicable"):
        return 0.0
    text = str(result.get("target", "")) + str(result.get("target_region", "")) + str(result)
    if target in text or target in {"slot", "field"}:
        return _generic_drop(result)
    return _generic_drop(result)
