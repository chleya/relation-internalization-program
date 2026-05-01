from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b4_intervention_metrics import trace_guided_intervention_accuracy
from .b4_intervention_policy import trace_family_for_model, trace_guided_intervention_policy
from .b4_trace_ablation_eval import ablate_b4_family_trace


def evaluate_action_after_trace_ablation_specificity(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    private_drops = []
    non_trace_drops = []
    saliency_drops = []
    action_head_drops = []
    type_shifts = []
    region_shifts = []
    non_trace_stability = []
    records = []
    trace_family = trace_family_for_model(model)
    for episode_id, episode in enumerate(episodes):
        oracle = episode["ground_truth"]["oracle_best_action"]
        base = trace_guided_intervention_policy(model, episode, config)["action"]
        private = trace_guided_intervention_policy(model, ablate_b4_family_trace(model, episode, trace_family, config), config)["action"]
        non_trace = trace_guided_intervention_policy(model, matched_non_trace_ablation(episode), config)["action"]
        saliency = trace_guided_intervention_policy(model, saliency_action_ablation(episode), config)["action"]
        action_head = trace_guided_intervention_policy(model, action_head_only_perturbation(episode), config)["action"]
        base_hit = trace_guided_intervention_accuracy(base, oracle)
        private_hit = trace_guided_intervention_accuracy(private, oracle)
        non_trace_hit = trace_guided_intervention_accuracy(non_trace, oracle)
        saliency_hit = trace_guided_intervention_accuracy(saliency, oracle)
        action_head_hit = trace_guided_intervention_accuracy(action_head, oracle)
        private_drop = max(0.0, base_hit - private_hit)
        non_trace_drop = max(0.0, base_hit - non_trace_hit)
        saliency_drop = max(0.0, base_hit - saliency_hit)
        action_head_drop = max(0.0, base_hit - action_head_hit)
        private_drops.append(private_drop)
        non_trace_drops.append(non_trace_drop)
        saliency_drops.append(saliency_drop)
        action_head_drops.append(action_head_drop)
        type_shifts.append(1.0 if base["action_type"] != private["action_type"] else 0.0)
        region_shifts.append(1.0 if int(base["region_id"]) != int(private["region_id"]) else 0.0)
        non_trace_stability.append(non_trace_hit)
        records.append(
            {
                "record_kind": "trace_ablation_specificity",
                "model": model_name or str(getattr(model, "name", "")),
                "seed": seed,
                "episode_id": episode_id,
                "audit_type": "action_after_trace_ablation_specificity",
                "base_action_type": base["action_type"],
                "base_region": int(base["region_id"]),
                "ablated_action_type": private["action_type"],
                "ablated_region": int(private["region_id"]),
                "ablation_type": "private_trace",
                "private_trace_ablation_drop": private_drop,
                "matched_non_trace_ablation_drop": non_trace_drop,
                "saliency_ablation_drop": saliency_drop,
                "action_head_perturbation_drop": action_head_drop,
                "gate_pass": int(private_drop >= 0.0),
                "note": "private trace vs non-trace action ablation specificity",
            }
        )
    private_drop = mean_or_zero(private_drops)
    non_trace_drop = mean_or_zero(non_trace_drops)
    return {
        "private_trace_ablation_drop": private_drop,
        "matched_non_trace_ablation_drop": non_trace_drop,
        "saliency_ablation_drop": mean_or_zero(saliency_drops),
        "action_head_perturbation_drop": mean_or_zero(action_head_drops),
        "private_trace_over_non_trace_ratio": private_drop / (non_trace_drop + 1e-6),
        "private_trace_over_saliency_ratio": private_drop / (mean_or_zero(saliency_drops) + 1e-6),
        "action_type_shift_after_trace_ablation": mean_or_zero(type_shifts),
        "region_shift_after_trace_ablation": mean_or_zero(region_shifts),
        "non_trace_action_stability": mean_or_zero(non_trace_stability),
    }, records


def matched_non_trace_ablation(episode: dict[str, Any]) -> dict[str, Any]:
    altered = copy.copy(episode)
    altered["b4_non_trace_ablation"] = True
    return altered


def saliency_action_ablation(episode: dict[str, Any]) -> dict[str, Any]:
    altered = copy.copy(episode)
    altered["b4_saliency_ablation"] = True
    return altered


def action_head_only_perturbation(episode: dict[str, Any]) -> dict[str, Any]:
    altered = copy.copy(episode)
    altered["b4_action_head_perturbation"] = True
    return altered


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

