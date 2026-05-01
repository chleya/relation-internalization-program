from __future__ import annotations

from typing import Any

import numpy as np

from .b3_active_inspection_baselines import evaluate_b3_baselines
from .b3_information_gain import compute_information_gain_after_inspection
from .b3_inspection_policy import trace_guided_inspection_policy


def run_b31_baseline_sanity_check(
    models: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    baseline_metrics, baseline_records = evaluate_b3_baselines(episodes, config, seed)
    trace_gains = []
    records = []
    first_model = next(iter(models.values())) if models else None
    for episode_id, episode in enumerate(episodes):
        if first_model is None:
            continue
        policy = trace_guided_inspection_policy(first_model, episode, config)
        gain = compute_information_gain_after_inspection(first_model, episode, int(policy["inspect_region"]), config)["absolute_gain"]
        trace_gains.append(gain)
        records.append(
            {
                "seed": seed,
                "episode_id": episode_id,
                "trace_inspect_region": int(policy["inspect_region"]),
                "trace_gain": gain,
                "oracle_best_inspect_region": int(episode["ground_truth"]["oracle_best_inspect_region"]),
            }
        )
    trace_gain = mean_or_zero(trace_gains)
    metrics = {
        "random_inspection_score": baseline_metrics["random_inspection_score"],
        "saliency_inspection_score": baseline_metrics["saliency_inspection_score"],
        "short_horizon_inspection_score": baseline_metrics["short_horizon_inspection_score"],
        "oracle_inspection_score": baseline_metrics["oracle_inspection_score"],
        "trace_over_saliency_gain_margin": trace_gain - baseline_metrics["saliency_information_gain"],
        "trace_over_short_horizon_gain_margin": trace_gain - baseline_metrics["short_horizon_information_gain"],
        "oracle_over_trace_gap": baseline_metrics["oracle_information_gain"] - trace_gain,
    }
    records.extend({"seed": seed, **row} for row in baseline_records)
    return metrics, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
