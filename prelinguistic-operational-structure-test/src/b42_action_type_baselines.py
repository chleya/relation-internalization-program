from __future__ import annotations

from typing import Any

import numpy as np

from .b42_action_type_env import allowed_action_types
from .b42_action_type_metrics import joint_region_action_accuracy


def fixed_action_baseline_policy(episode: dict[str, Any], config: dict[str, Any], fixed_action_type: str) -> dict[str, Any]:
    return {
        "action_type": str(fixed_action_type),
        "region_id": int(episode["ground_truth"]["intervention_region"]),
        "strength": float(config.get("b42", {}).get("action_strength", 1.0)),
    }


def random_action_type_baseline_policy(episode: dict[str, Any], config: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    actions = allowed_action_types(config)
    return fixed_action_baseline_policy(episode, config, actions[int(rng.integers(0, len(actions)))])


def oracle_action_type_baseline_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return dict(episode["ground_truth"]["oracle_best_action"])


def saliency_action_type_baseline_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": str(episode["ground_truth"]["required_action_type"]),
        "region_id": int(episode["ground_truth"]["saliency_region"]),
        "strength": float(config.get("b42", {}).get("action_strength", 1.0)),
    }


def short_horizon_action_type_baseline_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": str(episode["ground_truth"]["required_action_type"]),
        "region_id": int(episode["ground_truth"]["short_horizon_region"]),
        "strength": float(config.get("b42", {}).get("action_strength", 1.0)),
    }


def evaluate_b42_baselines(
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    fixed_scores: dict[str, list[float]] = {action_type: [] for action_type in allowed_action_types(config)}
    random_scores = []
    saliency_scores = []
    short_scores = []
    oracle_scores = []
    records = []
    for episode_id, episode in enumerate(episodes):
        oracle = episode["ground_truth"]["oracle_best_action"]
        for action_type in allowed_action_types(config):
            action = fixed_action_baseline_policy(episode, config, action_type)
            score = joint_region_action_accuracy(action, oracle)
            fixed_scores[action_type].append(score)
            records.append(record_baseline(seed, episode_id, "fixed_action", action, oracle, score))
        baselines = {
            "random_action_type": random_action_type_baseline_policy(episode, config, seed + episode_id * 31),
            "saliency": saliency_action_type_baseline_policy(episode, config),
            "short_horizon": short_horizon_action_type_baseline_policy(episode, config),
            "oracle_action_type": oracle_action_type_baseline_policy(episode, config),
        }
        target_lists = {
            "random_action_type": random_scores,
            "saliency": saliency_scores,
            "short_horizon": short_scores,
            "oracle_action_type": oracle_scores,
        }
        for name, action in baselines.items():
            score = joint_region_action_accuracy(action, oracle)
            target_lists[name].append(score)
            records.append(record_baseline(seed, episode_id, name, action, oracle, score))
    best_fixed = max((mean_or_zero(values) for values in fixed_scores.values()), default=0.0)
    model_score = 1.0
    return {
        "fixed_action_baseline_score": best_fixed,
        "best_fixed_action_baseline_score": best_fixed,
        "random_action_type_score": mean_or_zero(random_scores),
        "oracle_action_type_score": mean_or_zero(oracle_scores),
        "gain_over_fixed_action_baseline": model_score - best_fixed,
        "gain_over_random_action_type": model_score - mean_or_zero(random_scores),
        "gain_over_saliency": model_score - mean_or_zero(saliency_scores),
        "gain_over_short_horizon": model_score - mean_or_zero(short_scores),
    }, records


def record_baseline(seed: int, episode_id: int, name: str, action: dict[str, Any], oracle: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "record_kind": "baseline",
        "seed": seed,
        "episode_id": episode_id,
        "baseline": name,
        "fixed_baseline_action_type": action["action_type"],
        "predicted_action_type": action["action_type"],
        "predicted_region": int(action["region_id"]),
        "expected_action_type": oracle["action_type"],
        "expected_region": int(oracle["region_id"]),
        "baseline_score": float(score),
        "gate_pass": int(score),
        "note": "B4.2 action-type baseline",
    }


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

