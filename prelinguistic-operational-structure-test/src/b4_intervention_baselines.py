from __future__ import annotations

from typing import Any

import numpy as np

from .b4_action_space import enumerate_candidate_actions
from .b4_intervention_values import compute_intervention_value, compute_oracle_best_intervention
from .b4_intervention_metrics import trace_guided_intervention_accuracy


DEFAULT_ACTION = "stabilize_trace_region"


def random_intervention_policy(episode: dict[str, Any], config: dict[str, Any], seed: int) -> dict[str, Any]:
    actions = enumerate_candidate_actions(config)
    rng = np.random.default_rng(seed)
    return dict(actions[int(rng.integers(0, len(actions)))])


def saliency_intervention_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": DEFAULT_ACTION,
        "region_id": int(episode["ground_truth"]["saliency_region"]),
        "strength": float(config.get("b4", {}).get("action_strength", 1.0)),
    }


def short_horizon_intervention_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": DEFAULT_ACTION,
        "region_id": int(episode["ground_truth"]["short_horizon_region"]),
        "strength": float(config.get("b4", {}).get("action_strength", 1.0)),
    }


def inspect_only_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": "inspect_only",
        "region_id": int(episode["ground_truth"]["oracle_best_action"]["region_id"]),
        "strength": float(config.get("b4", {}).get("action_strength", 1.0)),
    }


def oracle_intervention_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return compute_oracle_best_intervention(episode, config)


def evaluate_b4_baselines(
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    random_scores = []
    saliency_scores = []
    short_scores = []
    inspect_scores = []
    oracle_scores = []
    records = []
    for episode_id, episode in enumerate(episodes):
        oracle = episode["ground_truth"]["oracle_best_action"]
        policies = {
            "random": random_intervention_policy(episode, config, seed + episode_id * 17),
            "saliency": saliency_intervention_policy(episode, config),
            "short_horizon": short_horizon_intervention_policy(episode, config),
            "inspect_only": inspect_only_policy(episode, config),
            "oracle": oracle_intervention_policy(episode, config),
        }
        score_lists = {
            "random": random_scores,
            "saliency": saliency_scores,
            "short_horizon": short_scores,
            "inspect_only": inspect_scores,
            "oracle": oracle_scores,
        }
        for name, action in policies.items():
            score = trace_guided_intervention_accuracy(action, oracle)
            value = compute_intervention_value(episode, action, config)
            score_lists[name].append(score)
            records.append(
                {
                    "record_kind": "baseline",
                    "seed": seed,
                    "episode_id": episode_id,
                    "baseline": name,
                    "predicted_action_type": action["action_type"],
                    "predicted_region": int(action["region_id"]),
                    "oracle_action_type": oracle["action_type"],
                    "oracle_region": int(oracle["region_id"]),
                    "baseline_score": score,
                    "intervention_value": value,
                }
            )
    trace_score = 1.0
    return {
        "random_intervention_score": mean_or_zero(random_scores),
        "saliency_intervention_score": mean_or_zero(saliency_scores),
        "short_horizon_intervention_score": mean_or_zero(short_scores),
        "inspect_only_score": mean_or_zero(inspect_scores),
        "oracle_intervention_score": mean_or_zero(oracle_scores),
        "gain_over_random": trace_score - mean_or_zero(random_scores),
        "gain_over_saliency": trace_score - mean_or_zero(saliency_scores),
        "gain_over_short_horizon": trace_score - mean_or_zero(short_scores),
        "gain_over_inspect_only": trace_score - mean_or_zero(inspect_scores),
        "intervention_vs_inspection_gain": trace_score - mean_or_zero(inspect_scores),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

