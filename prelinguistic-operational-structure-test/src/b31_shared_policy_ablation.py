from __future__ import annotations

from typing import Any

import numpy as np

from .b3_inspection_policy import trace_guided_inspection_policy


def disable_shared_inspection_policy(model: Any) -> Any:
    setattr(model, "shared_inspection_policy_disabled", True)
    return model


def evaluate_shared_inspection_policy_ablation(
    models: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    base_hits = []
    ablated_hits = []
    usage_after = []
    records = []
    for model_name, model in models.items():
        for episode_id, episode in enumerate(episodes):
            oracle = int(episode["ground_truth"]["oracle_best_inspect_region"])
            base_policy = trace_guided_inspection_policy(model, episode, config)
            disable_shared_inspection_policy(model)
            ablated_policy = trace_guided_inspection_policy(model, episode, config)
            base_hit = 1.0 if int(base_policy["inspect_region"]) == oracle else 0.0
            ablated_hit = 1.0 if int(ablated_policy["inspect_region"]) == oracle else 0.0
            shared_after = bool(ablated_policy.get("provenance", {}).get("shared_inspection_policy_used", False))
            base_hits.append(base_hit)
            ablated_hits.append(ablated_hit)
            usage_after.append(1.0 if shared_after else 0.0)
            records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "episode_id": episode_id,
                    "base_region": int(base_policy["inspect_region"]),
                    "after_shared_ablation_region": int(ablated_policy["inspect_region"]),
                    "oracle_best_inspect_region": oracle,
                    "base_correct": base_hit,
                    "after_shared_ablation_correct": ablated_hit,
                    "shared_policy_used_after_ablation": int(shared_after),
                }
            )
    base_score = mean_or_zero(base_hits)
    ablated_score = mean_or_zero(ablated_hits)
    return {
        "shared_policy_ablation_drop": max(0.0, base_score - ablated_score),
        "private_inspection_retention_after_shared_ablation": float(ablated_score / (base_score + 1e-6)) if base_score > 0.0 else 0.0,
        "shared_policy_usage_rate_after_ablation": mean_or_zero(usage_after),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
