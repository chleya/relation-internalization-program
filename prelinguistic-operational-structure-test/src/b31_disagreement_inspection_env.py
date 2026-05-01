from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b22_disagreement_env import make_trace_disagreement_episode
from .b3_inspection_policy import trace_guided_inspection_policy


CAUSAL_TO_B22 = {
    "recurrent_memory": "recurrent",
    "field_trace": "field",
    "schema_memory": "schema",
}

FAMILY_REGION_FIELD = {
    "recurrent_flow_checkpoint_model": "recurrent_trace_region",
    "field_memory_model": "field_trace_region",
    "schema_memory_model": "schema_trace_region",
}


def make_disagreement_inspection_episode(config: dict[str, Any], seed: int, causal_family: str) -> dict[str, Any]:
    b22_family = CAUSAL_TO_B22.get(str(causal_family), str(causal_family))
    episode = copy.deepcopy(make_trace_disagreement_episode(config, seed, b22_family))
    gt = episode["ground_truth"]
    recurrent_region = int(gt["temporal_trace_region"])
    field_region = int(gt["field_trace_region"])
    schema_region = int(gt["schema_trace_region"])
    causal_field = {
        "recurrent_memory": recurrent_region,
        "field_trace": field_region,
        "schema_memory": schema_region,
    }.get(str(causal_family), int(gt["causal_region"]))
    saliency_region = int(gt.get("early_saliency_region", gt.get("saliency_region", 0)))
    short_horizon_region = saliency_region
    values = {idx: 0.0 for idx in range(int(config.get("env", {}).get("grid_size", 8)) ** 2)}
    values[causal_field] = 1.0
    if saliency_region != causal_field:
        values[saliency_region] = 0.05
    gt.update(
        {
            "episode_type": "b31_disagreement_inspection",
            "recurrent_trace_region": recurrent_region,
            "field_trace_region": field_region,
            "schema_trace_region": schema_region,
            "saliency_region": saliency_region,
            "short_horizon_region": short_horizon_region,
            "causal_family": str(causal_family),
            "oracle_best_inspect_region": int(causal_field),
            "true_trace_region": int(causal_field),
            "inspection_values": values,
        }
    )
    return episode


def evaluate_disagreement_inspection_divergence(
    models: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    divergences = []
    exact_matches = []
    family_hits = []
    causal_hits = []
    records = []
    for episode_id, episode in enumerate(episodes):
        gt = episode["ground_truth"]
        predictions = {}
        for model_name, model in models.items():
            policy = trace_guided_inspection_policy(model, episode, config)
            pred = int(policy["inspect_region"])
            predictions[model_name] = pred
            family_field = FAMILY_REGION_FIELD.get(model_name, "")
            if family_field:
                family_hits.append(1.0 if pred == int(gt[family_field]) else 0.0)
            causal_hits.append(1.0 if pred == int(gt["oracle_best_inspect_region"]) else 0.0)
            records.append(
                {
                    "seed": seed,
                    "episode_id": episode_id,
                    "model": model_name,
                    "predicted_inspect_region": pred,
                    "recurrent_trace_region": int(gt["recurrent_trace_region"]),
                    "field_trace_region": int(gt["field_trace_region"]),
                    "schema_trace_region": int(gt["schema_trace_region"]),
                    "saliency_region": int(gt["saliency_region"]),
                    "short_horizon_region": int(gt["short_horizon_region"]),
                    "causal_family": gt["causal_family"],
                    "oracle_best_inspect_region": int(gt["oracle_best_inspect_region"]),
                }
            )
        all_same = len(set(predictions.values())) <= 1
        exact_matches.append(1.0 if all_same else 0.0)
        divergences.append(0.0 if all_same else 1.0)
    return {
        "disagreement_inspection_divergence": mean_or_zero(divergences),
        "family_aligned_inspection_rate": mean_or_zero(family_hits),
        "cross_model_same_inspect_rate_on_disagreement": mean_or_zero(exact_matches),
        "causal_family_inspection_accuracy": mean_or_zero(causal_hits),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
