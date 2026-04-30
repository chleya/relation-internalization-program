from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b2_delayed_env import choose_far_region, draw_trace, make_delayed_checkpoint_episode
from .inspect_policy import select_region_from_logits
from .model_io import make_model_batch


FAMILY_FIELD = {
    "recurrent_flow_checkpoint": "temporal_trace_region",
    "field_memory": "field_trace_region",
    "schema_memory": "schema_trace_region",
}

CAUSAL_FAMILY_TO_TRACE_FAMILY = {
    "recurrent": "recurrent_flow_checkpoint",
    "temporal": "recurrent_flow_checkpoint",
    "field": "field_memory",
    "schema": "schema_memory",
    "recurrent_flow_checkpoint": "recurrent_flow_checkpoint",
    "field_memory": "field_memory",
    "schema_memory": "schema_memory",
}


def make_trace_disagreement_episode(config: dict[str, Any], seed: int, causal_family: str) -> dict[str, Any]:
    trace_family = CAUSAL_FAMILY_TO_TRACE_FAMILY.get(str(causal_family), str(causal_family))
    episode = make_delayed_checkpoint_episode(config, seed, delay=4, saliency_decoy=True)
    altered = copy.deepcopy(episode)
    env = config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    causal_region = int(altered["ground_truth"]["true_delayed_checkpoint_region"])

    regions = {
        "recurrent_flow_checkpoint": choose_far_region(causal_region, grid_size, seed + 11),
        "field_memory": choose_far_region(causal_region, grid_size, seed + 23),
        "schema_memory": choose_far_region(causal_region, grid_size, seed + 37),
    }
    regions[trace_family] = causal_region
    ensure_distinct_regions(regions, causal_region, grid_size, seed)

    for frame in altered["past_frames"]:
        draw_trace(frame, regions["recurrent_flow_checkpoint"], frame_size, grid_size, 1.00)
        draw_trace(frame, regions["field_memory"], frame_size, grid_size, 0.82)
        draw_trace(frame, regions["schema_memory"], frame_size, grid_size, 0.64)
    altered["frames"][: len(altered["past_frames"])] = altered["past_frames"]
    altered["ground_truth"].update(
        {
            "episode_type": "b22_trace_disagreement",
            "temporal_trace_region": int(regions["recurrent_flow_checkpoint"]),
            "field_trace_region": int(regions["field_memory"]),
            "schema_trace_region": int(regions["schema_memory"]),
            "causal_family": str(causal_family),
            "causal_trace_family": trace_family,
            "causal_region": int(causal_region),
            "critical_inspection_region": int(causal_region),
            "true_delayed_checkpoint_region": int(causal_region),
        }
    )
    return altered


def evaluate_disagreement_episode_divergence(models: dict[str, Any], episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    if not models or not episodes:
        return {
            "disagreement_episode_divergence": 0.0,
            "family_aligned_selection_rate": 0.0,
            "cross_model_exact_prediction_match_rate_on_disagreement": 1.0,
            "causal_family_accuracy": 0.0,
        }
    divergences = []
    exact_matches = []
    family_aligned = []
    causal_hits = []
    for episode in episodes:
        gt = episode["ground_truth"]
        predictions: dict[str, int] = {}
        for model_name, model in models.items():
            batch = make_model_batch(episode, config)
            predictions[model_name] = select_region_from_logits(model.forward(batch).get("inspection_logits"))
        values = list(predictions.values())
        all_same = len(set(values)) <= 1
        divergences.append(0.0 if all_same else 1.0)
        exact_matches.append(1.0 if all_same else 0.0)
        for model_name, pred in predictions.items():
            field_name = expected_region_field(model_name)
            if field_name:
                family_aligned.append(1.0 if int(pred) == int(gt[field_name]) else 0.0)
            causal_hits.append(1.0 if int(pred) == int(gt["causal_region"]) else 0.0)
    return {
        "disagreement_episode_divergence": float(np.mean(divergences)),
        "family_aligned_selection_rate": float(np.mean(family_aligned)) if family_aligned else 0.0,
        "cross_model_exact_prediction_match_rate_on_disagreement": float(np.mean(exact_matches)),
        "causal_family_accuracy": float(np.mean(causal_hits)) if causal_hits else 0.0,
    }


def ensure_distinct_regions(regions: dict[str, int], causal_region: int, grid_size: int, seed: int) -> None:
    used: set[int] = set()
    for idx, family in enumerate(["recurrent_flow_checkpoint", "field_memory", "schema_memory"]):
        region = int(regions[family])
        attempts = 0
        while region in used and attempts < grid_size * grid_size:
            region = choose_far_region(causal_region + attempts + idx, grid_size, seed + attempts + idx * 101)
            attempts += 1
        regions[family] = int(region)
        used.add(int(region))


def expected_region_field(model_name: str) -> str:
    if "recurrent_flow_checkpoint" in model_name:
        return "temporal_trace_region"
    if "field_memory" in model_name:
        return "field_trace_region"
    if "schema_memory" in model_name:
        return "schema_trace_region"
    return ""
