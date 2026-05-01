from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b4_intervention_env import canonical_family
from .b4_intervention_metrics import trace_guided_intervention_accuracy
from .b4_intervention_policy import trace_family_for_model, trace_guided_intervention_policy


def ablate_b4_family_trace(model: Any, episode: dict[str, Any], family: str, config: dict[str, Any]) -> dict[str, Any]:
    altered = copy.copy(episode)
    altered["ground_truth"] = dict(episode.get("ground_truth", {}))
    family_key = canonical_family(family)
    target = int(altered["ground_truth"][f"{family_key}_intervention_region"])
    altered["b4_ablation"] = {
        "family": family_key,
        "target_region": target,
        "ablation_type": "family_specific_intervention_score_suppression",
    }
    return altered


def evaluate_action_after_trace_ablation(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    base_hits = []
    ablated_hits = []
    non_target_hits = []
    type_shifts = []
    region_shifts = []
    records = []
    trace_family = trace_family_for_model(model)
    for episode_id, episode in enumerate(episodes):
        oracle = episode["ground_truth"]["oracle_best_action"]
        base = trace_guided_intervention_policy(model, episode, config)["action"]
        ablated_episode = ablate_b4_family_trace(model, episode, trace_family, config)
        ablated = trace_guided_intervention_policy(model, ablated_episode, config)["action"]
        base_hit = trace_guided_intervention_accuracy(base, oracle)
        ablated_hit = trace_guided_intervention_accuracy(ablated, oracle)
        base_hits.append(base_hit)
        ablated_hits.append(ablated_hit)
        type_shifts.append(1.0 if base["action_type"] != ablated["action_type"] else 0.0)
        region_shifts.append(1.0 if int(base["region_id"]) != int(ablated["region_id"]) else 0.0)
        for non_target_family in ["recurrent", "field", "schema"]:
            if non_target_family == trace_family:
                continue
            non_target_episode = ablate_b4_family_trace(model, episode, non_target_family, config)
            non_target_action = trace_guided_intervention_policy(model, non_target_episode, config)["action"]
            non_target_hits.append(trace_guided_intervention_accuracy(non_target_action, oracle))
        drop = max(0.0, base_hit - ablated_hit)
        records.append(
            {
                "record_kind": "trace_ablation",
                "model": model_name or str(getattr(model, "name", "")),
                "seed": seed,
                "episode_id": episode_id,
                "family": trace_family,
                "predicted_action_type": base["action_type"],
                "predicted_region": int(base["region_id"]),
                "trace_ablated_action_type": ablated["action_type"],
                "trace_ablated_region": int(ablated["region_id"]),
                "trace_ablation_drop": drop,
                "gate_pass": int(drop >= 0.0),
                "note": "action after private trace ablation",
            }
        )
    base_acc = mean_or_zero(base_hits)
    ablated_acc = mean_or_zero(ablated_hits)
    return {
        "base_intervention_accuracy": base_acc,
        "trace_ablated_intervention_accuracy": ablated_acc,
        "trace_ablation_intervention_drop": max(0.0, base_acc - ablated_acc),
        "non_trace_action_stability": mean_or_zero(non_target_hits),
        "action_type_shift_after_ablation": mean_or_zero(type_shifts),
        "region_shift_after_ablation": mean_or_zero(region_shifts),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

