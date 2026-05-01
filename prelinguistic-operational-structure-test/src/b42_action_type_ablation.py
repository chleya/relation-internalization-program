from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b42_action_type_metrics import joint_region_action_accuracy, region_accuracy
from .b42_action_type_policy import action_type_disambiguating_policy


def ablate_action_type_scorer(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    altered = copy.copy(episode)
    altered["b42_action_type_ablation"] = True
    altered["b42_trace_action_ablation"] = True
    altered["b42_ablation_fallback_action"] = config.get("b42", {}).get("allowed_action_types", ["apply_local_damping"])[0]
    return altered


def mask_action_type(model: Any, episode: dict[str, Any], masked_action_type: str, config: dict[str, Any]) -> dict[str, Any]:
    altered = copy.copy(episode)
    altered["b42_masked_action_type"] = str(masked_action_type)
    return altered


def evaluate_action_type_ablation(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    base_hits = []
    ablated_hits = []
    shifts = []
    region_stability = []
    non_target_stability = []
    records = []
    for episode_id, episode in enumerate(episodes):
        expected = episode["ground_truth"]["oracle_best_action"]
        base = action_type_disambiguating_policy(model, episode, config)["action"]
        ablated = action_type_disambiguating_policy(model, ablate_action_type_scorer(model, episode, config), config)["action"]
        base_hit = joint_region_action_accuracy(base, expected)
        ablated_hit = joint_region_action_accuracy(ablated, expected)
        base_hits.append(base_hit)
        ablated_hits.append(ablated_hit)
        shifts.append(1.0 if base["action_type"] != ablated["action_type"] else 0.0)
        region_stability.append(region_accuracy(ablated, expected))
        masked = action_type_disambiguating_policy(model, mask_action_type(model, episode, "do_not_mask_real_action", config), config)["action"]
        non_target_stability.append(joint_region_action_accuracy(masked, expected))
        records.append(
            {
                "record_kind": "action_type_ablation",
                "model": model_name or str(getattr(model, "name", "")),
                "seed": seed,
                "episode_id": episode_id,
                "ablation_type": "action_type_scorer",
                "base_action_type": base["action_type"],
                "ablated_action_type": ablated["action_type"],
                "predicted_region": int(base["region_id"]),
                "expected_region": int(expected["region_id"]),
                "expected_action_type": expected["action_type"],
                "gate_pass": int(base_hit),
                "note": "action-type scorer ablation preserving region scoring",
            }
        )
    base_acc = mean_or_zero(base_hits)
    ablated_acc = mean_or_zero(ablated_hits)
    return {
        "action_type_ablation_drop": max(0.0, base_acc - ablated_acc),
        "action_type_shift_after_ablation": mean_or_zero(shifts),
        "action_type_shift_after_trace_ablation": mean_or_zero(shifts),
        "region_stability_after_action_ablation": mean_or_zero(region_stability),
        "non_target_action_stability": mean_or_zero(non_target_stability),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

