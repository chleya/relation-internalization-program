from __future__ import annotations

from typing import Any

import numpy as np

from .b2_delayed_env import choose_far_region
from .b3_inspection_policy import trace_guided_inspection_policy
from .b3_trace_ablation_eval import ablate_episode_region


def evaluate_trace_ablation_specificity(
    models: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    private_drops = []
    non_trace_drops = []
    saliency_drops = []
    non_trace_stability = []
    records = []
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    for model_name, model in models.items():
        for episode_id, episode in enumerate(episodes):
            oracle = int(episode["ground_truth"]["oracle_best_inspect_region"])
            true_region = int(episode["ground_truth"]["true_trace_region"])
            saliency_region = int(episode["ground_truth"]["saliency_region"])
            non_trace_region = choose_far_region(true_region + saliency_region, grid_size, seed + episode_id + 1000)
            base_region = int(trace_guided_inspection_policy(model, episode, config)["inspect_region"])
            trace_region = int(trace_guided_inspection_policy(model, ablate_episode_region(episode, true_region, config), config)["inspect_region"])
            non_trace_region_after = int(
                trace_guided_inspection_policy(model, ablate_episode_region(episode, non_trace_region, config), config)["inspect_region"]
            )
            saliency_region_after = int(
                trace_guided_inspection_policy(model, ablate_episode_region(episode, saliency_region, config), config)["inspect_region"]
            )
            base_hit = 1.0 if base_region == oracle else 0.0
            trace_hit = 1.0 if trace_region == oracle else 0.0
            non_trace_hit = 1.0 if non_trace_region_after == oracle else 0.0
            saliency_hit = 1.0 if saliency_region_after == oracle else 0.0
            private_drop = max(0.0, base_hit - trace_hit)
            non_trace_drop = max(0.0, base_hit - non_trace_hit)
            saliency_drop = max(0.0, base_hit - saliency_hit)
            private_drops.append(private_drop)
            non_trace_drops.append(non_trace_drop)
            saliency_drops.append(saliency_drop)
            non_trace_stability.append(non_trace_hit)
            records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "episode_id": episode_id,
                    "base_region": base_region,
                    "trace_ablated_region": trace_region,
                    "matched_non_trace_region": non_trace_region_after,
                    "saliency_ablated_region": saliency_region_after,
                    "oracle_best_inspect_region": oracle,
                    "private_trace_ablation_drop": private_drop,
                    "matched_non_trace_ablation_drop": non_trace_drop,
                    "saliency_ablation_drop": saliency_drop,
                }
            )
    private = mean_or_zero(private_drops)
    non_trace = mean_or_zero(non_trace_drops)
    saliency = mean_or_zero(saliency_drops)
    return {
        "private_trace_ablation_drop": private,
        "matched_non_trace_ablation_drop": non_trace,
        "saliency_ablation_drop": saliency,
        "private_trace_over_non_trace_ratio": float(private / (non_trace + 1e-6)),
        "private_trace_over_saliency_ratio": float(private / (saliency + 1e-6)),
        "non_trace_inspection_stability": mean_or_zero(non_trace_stability),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
