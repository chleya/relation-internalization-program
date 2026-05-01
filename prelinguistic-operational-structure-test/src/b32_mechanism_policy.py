from __future__ import annotations

from typing import Any

import numpy as np

from .b31_inspection_scorers import field_trace_inspection_scores, recurrent_trace_inspection_scores, schema_trace_inspection_scores
from .b32_goal_conditioning import expected_region_for_goal, goal_family_from_code
from .b32_inspection_values import best_region, compute_combined_inspection_value


MODEL_TO_NATIVE_GOAL = {
    "recurrent_flow_checkpoint_model": "recurrent_goal",
    "field_memory_model": "field_goal",
    "schema_memory_model": "schema_goal",
}

GOAL_TO_SCORE_KEY = {
    "recurrent_goal": "recurrent",
    "field_goal": "field",
    "schema_goal": "schema",
}


def mechanism_conditioned_inspection_policy(
    model: Any,
    episode: dict[str, Any],
    goal_code: list[float],
    config: dict[str, Any],
) -> dict[str, Any]:
    goal_family = goal_family_from_code(goal_code)
    family_scores = cached_family_scores(model, episode, config)
    selected_scores = apply_family_ablation_hook(dict(family_scores[GOAL_TO_SCORE_KEY[goal_family]]), episode, goal_family)
    region = best_region(selected_scores)
    return {
        "inspect_region": int(region),
        "goal_code": [float(value) for value in goal_code],
        "goal_family": goal_family,
        "trace_family": GOAL_TO_SCORE_KEY[goal_family],
        "policy_source": f"{GOAL_TO_SCORE_KEY[goal_family]}_mechanism_inspection",
        "family_scores": family_scores,
        "selected_family_score": float(selected_scores.get(region, 0.0)),
        "combined_value": compute_combined_inspection_value(episode, region, goal_code, config),
        "provenance": {
            "shared_inspection_policy_used": False,
            "private_trace_inspection_score_used": True,
            "oracle_region_used": False,
            "family_ablation_hook_used": bool(episode.get("b32_ablation")),
        },
    }


def apply_family_ablation_hook(scores: dict[int, float], episode: dict[str, Any], goal_family: str) -> dict[int, float]:
    ablation = episode.get("b32_ablation", {})
    if ablation.get("family") != goal_family or not scores:
        return scores
    target = int(ablation.get("target_region", expected_region_for_goal(episode, goal_family)))
    floor = min(float(value) for value in scores.values()) - 1.0
    scores[target] = floor
    return scores


def cached_family_scores(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, dict[int, float]]:
    cache_key = "_b32_family_scores_cache"
    if cache_key not in episode:
        episode[cache_key] = {
            "recurrent": recurrent_trace_inspection_scores(model, episode, config),
            "field": field_trace_inspection_scores(model, episode, config),
            "schema": schema_trace_inspection_scores(model, episode, config),
        }
    return episode[cache_key]


def evaluate_mechanism_conditioned_policy(
    models: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records = []
    family_hits = []
    recurrent_hits = []
    field_hits = []
    schema_hits = []
    switch_hits = []
    value_alignment = []
    predictions_by_episode: dict[int, dict[str, int]] = {}
    for episode_id, episode in enumerate(episodes):
        for model_name, model in models.items():
            model_predictions = {}
            for goal_family, goal_code in [
                ("recurrent_goal", [1.0, 0.0, 0.0]),
                ("field_goal", [0.0, 1.0, 0.0]),
                ("schema_goal", [0.0, 0.0, 1.0]),
            ]:
                policy = mechanism_conditioned_inspection_policy(model, episode, goal_code, config)
                pred = int(policy["inspect_region"])
                expected = expected_region_for_goal(episode, goal_family)
                hit = 1.0 if pred == expected else 0.0
                family_hits.append(hit)
                value_alignment.append(hit)
                if goal_family == "recurrent_goal":
                    recurrent_hits.append(hit)
                elif goal_family == "field_goal":
                    field_hits.append(hit)
                else:
                    schema_hits.append(hit)
                model_predictions[goal_family] = pred
                records.append(record_from_policy(seed, episode_id, model_name, episode, goal_family, expected, policy, hit))
            expected_by_goal = {
                "recurrent_goal": int(episode["ground_truth"]["recurrent_inspect_region"]),
                "field_goal": int(episode["ground_truth"]["field_inspect_region"]),
                "schema_goal": int(episode["ground_truth"]["schema_inspect_region"]),
            }
            switch_hits.append(1.0 if model_predictions == expected_by_goal else 0.0)
            native_goal = MODEL_TO_NATIVE_GOAL.get(model_name, "recurrent_goal")
            predictions_by_episode.setdefault(episode_id, {})[model_name] = model_predictions[native_goal]
    disagreement_metrics = model_disagreement_metrics(predictions_by_episode)
    return {
        "family_specific_inspection_accuracy": mean_or_zero(family_hits),
        "recurrent_goal_accuracy": mean_or_zero(recurrent_hits),
        "field_goal_accuracy": mean_or_zero(field_hits),
        "schema_goal_accuracy": mean_or_zero(schema_hits),
        "task_conditioned_switch_accuracy": mean_or_zero(switch_hits),
        "inspect_value_decomposition_alignment": mean_or_zero(value_alignment),
        "recurrent_value_alignment": mean_or_zero(recurrent_hits),
        "field_value_alignment": mean_or_zero(field_hits),
        "schema_value_alignment": mean_or_zero(schema_hits),
        **disagreement_metrics,
    }, records


def evaluate_mechanism_disagreement(
    models: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    predictions_by_episode: dict[int, dict[str, int]] = {}
    records = []
    family_hits = []
    for episode_id, episode in enumerate(episodes):
        for model_name, model in models.items():
            native_goal = MODEL_TO_NATIVE_GOAL.get(model_name, "recurrent_goal")
            goal_code = {
                "recurrent_goal": [1.0, 0.0, 0.0],
                "field_goal": [0.0, 1.0, 0.0],
                "schema_goal": [0.0, 0.0, 1.0],
            }[native_goal]
            policy = mechanism_conditioned_inspection_policy(model, episode, goal_code, config)
            pred = int(policy["inspect_region"])
            expected = expected_region_for_goal(episode, native_goal)
            hit = 1.0 if pred == expected else 0.0
            family_hits.append(hit)
            predictions_by_episode.setdefault(episode_id, {})[model_name] = pred
            records.append(record_from_policy(seed, episode_id, model_name, episode, native_goal, expected, policy, hit))
    metrics = model_disagreement_metrics(predictions_by_episode)
    metrics["family_aligned_inspection_rate"] = mean_or_zero(family_hits)
    return metrics, records


def record_from_policy(
    seed: int,
    episode_id: int,
    model_name: str,
    episode: dict[str, Any],
    goal_family: str,
    expected: int,
    policy: dict[str, Any],
    hit: float,
) -> dict[str, Any]:
    gt = episode["ground_truth"]
    pred = int(policy["inspect_region"])
    values = gt["inspection_values"]
    return {
        "seed": seed,
        "model": model_name,
        "episode_id": episode_id,
        "goal_family": goal_family,
        "goal_code": " ".join(str(float(v)) for v in policy["goal_code"]),
        "predicted_inspect_region": pred,
        "expected_family_region": int(expected),
        "recurrent_inspect_region": int(gt["recurrent_inspect_region"]),
        "field_inspect_region": int(gt["field_inspect_region"]),
        "schema_inspect_region": int(gt["schema_inspect_region"]),
        "saliency_region": int(gt["saliency_region"]),
        "short_horizon_region": int(gt["short_horizon_region"]),
        "recurrent_value": float(values["recurrent_value"].get(pred, 0.0)),
        "field_value": float(values["field_value"].get(pred, 0.0)),
        "schema_value": float(values["schema_value"].get(pred, 0.0)),
        "combined_value": float(policy.get("combined_value", 0.0)),
        "gate_pass": int(hit),
        "note": "mechanism-conditioned inspection",
    }


def model_disagreement_metrics(predictions_by_episode: dict[int, dict[str, int]]) -> dict[str, float]:
    disagreement = []
    same = []
    entropies = []
    for predictions in predictions_by_episode.values():
        values = list(predictions.values())
        all_same = len(set(values)) <= 1
        same.append(1.0 if all_same else 0.0)
        disagreement.append(0.0 if all_same else 1.0)
        entropies.append(choice_entropy(values))
    return {
        "mechanism_disagreement_rate": mean_or_zero(disagreement),
        "cross_model_same_region_rate": mean_or_zero(same),
        "mechanism_specific_choice_entropy": mean_or_zero(entropies),
    }


def choice_entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    probs = [count / len(values) for count in counts.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0.0))


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
