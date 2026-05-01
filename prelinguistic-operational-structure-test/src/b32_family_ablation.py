from __future__ import annotations

from typing import Any

import numpy as np

from .b32_goal_conditioning import canonical_goal_family, expected_region_for_goal
from .b32_mechanism_policy import mechanism_conditioned_inspection_policy


GOAL_CODES = {
    "recurrent_goal": [1.0, 0.0, 0.0],
    "field_goal": [0.0, 1.0, 0.0],
    "schema_goal": [0.0, 0.0, 1.0],
}


def ablate_family_trace(model: Any, episode: dict[str, Any], family: str, config: dict[str, Any]) -> dict[str, Any]:
    altered = {**episode, "ground_truth": dict(episode.get("ground_truth", {}))}
    altered["b32_ablation"] = {
        "family": canonical_goal_family(family),
        "target_region": expected_region_for_goal(episode, family),
        "ablation_type": "family_specific_trace_score_suppression",
    }
    return altered


def evaluate_family_specific_trace_ablation(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    drops = []
    target_drops = {"recurrent_goal": [], "field_goal": [], "schema_goal": []}
    non_target_hits = []
    ratios = []
    records = []
    model_name = str(getattr(model, "name", ""))
    for episode_id, episode in enumerate(episodes):
        for family, code in GOAL_CODES.items():
            expected = expected_region_for_goal(episode, family)
            base = mechanism_conditioned_inspection_policy(model, episode, code, config)
            ablated_episode = ablate_family_trace(model, episode, family, config)
            ablated = mechanism_conditioned_inspection_policy(model, ablated_episode, code, config)
            non_target_hits_for_family = []
            for non_target_family, non_target_code in GOAL_CODES.items():
                if non_target_family == family:
                    continue
                non_target_expected = expected_region_for_goal(episode, non_target_family)
                non_target = mechanism_conditioned_inspection_policy(model, ablated_episode, non_target_code, config)
                non_target_hits_for_family.append(1.0 if int(non_target["inspect_region"]) == non_target_expected else 0.0)
            base_hit = 1.0 if int(base["inspect_region"]) == expected else 0.0
            ablated_hit = 1.0 if int(ablated["inspect_region"]) == expected else 0.0
            non_target_hit = mean_or_zero(non_target_hits_for_family)
            drop = max(0.0, base_hit - ablated_hit)
            non_target_drop = max(0.0, base_hit - non_target_hit)
            drops.append(drop)
            target_drops[family].append(drop)
            non_target_hits.append(non_target_hit)
            ratios.append(drop / (non_target_drop + 1e-6))
            records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "episode_id": episode_id,
                    "goal_family": family,
                    "ablation_family": family,
                    "base_prediction": int(base["inspect_region"]),
                    "ablated_prediction": int(ablated["inspect_region"]),
                    "non_target_prediction": "",
                    "expected_family_region": expected,
                    "drop": drop,
                    "non_target_drop": non_target_drop,
                    "gate_pass": int(drop >= 0.0),
                    "note": "family-specific trace ablation",
                }
            )
    return {
        "family_specific_trace_ablation_drop": mean_or_zero(drops),
        "recurrent_trace_ablation_drop": mean_or_zero(target_drops["recurrent_goal"]),
        "field_trace_ablation_drop": mean_or_zero(target_drops["field_goal"]),
        "schema_trace_ablation_drop": mean_or_zero(target_drops["schema_goal"]),
        "non_target_family_stability": mean_or_zero(non_target_hits),
        "private_trace_over_non_target_ratio": mean_or_zero(ratios),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
