from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b3_active_inspection_metrics import trace_ablation_inspection_drop
from .b3_inspection_policy import trace_guided_inspection_policy
from .features import region_id_to_slice


def evaluate_inspection_after_trace_ablation(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    base_hits = []
    ablated_hits = []
    control_hits = []
    records = []
    for idx, episode in enumerate(episodes):
        oracle = int(episode["ground_truth"]["oracle_best_inspect_region"])
        base_region = int(trace_guided_inspection_policy(model, episode, config)["inspect_region"])
        trace_ablated = ablate_episode_region(episode, int(episode["ground_truth"]["true_trace_region"]), config)
        control_ablated = ablate_episode_region(episode, int(episode["ground_truth"]["saliency_region"]), config)
        trace_region = int(trace_guided_inspection_policy(model, trace_ablated, config)["inspect_region"])
        control_region = int(trace_guided_inspection_policy(model, control_ablated, config)["inspect_region"])
        base_hit = 1.0 if base_region == oracle else 0.0
        ablated_hit = 1.0 if trace_region == oracle else 0.0
        control_hit = 1.0 if control_region == oracle else 0.0
        base_hits.append(base_hit)
        ablated_hits.append(ablated_hit)
        control_hits.append(control_hit)
        records.append(
            {
                "episode_id": idx,
                "base_region": base_region,
                "trace_ablated_region": trace_region,
                "control_ablated_region": control_region,
                "oracle_best_inspect_region": oracle,
                "base_correct": base_hit,
                "trace_ablated_correct": ablated_hit,
                "control_ablated_correct": control_hit,
            }
        )
    base_acc = mean_or_zero(base_hits)
    ablated_acc = mean_or_zero(ablated_hits)
    control_acc = mean_or_zero(control_hits)
    return {
        "base_trace_guided_inspection_accuracy": base_acc,
        "trace_ablated_inspection_accuracy": ablated_acc,
        "trace_ablation_inspection_drop": trace_ablation_inspection_drop(base_acc, ablated_acc),
        "non_trace_inspection_stability": control_acc,
    }, records


def ablate_episode_region(episode: dict[str, Any], region: int, config: dict[str, Any]) -> dict[str, Any]:
    altered = copy.deepcopy(episode)
    env = config.get("env", {})
    frame_size = int(env.get("frame_size", altered["past_frames"].shape[1]))
    grid_size = int(env.get("grid_size", 8))
    ys, xs = region_id_to_slice(int(region), frame_size, grid_size)
    past = np.asarray(altered["past_frames"], dtype=np.float32).copy()
    past[:, ys, xs, :] = 0.0
    altered["past_frames"] = past
    altered["frames"][: len(past)] = past
    return altered


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
