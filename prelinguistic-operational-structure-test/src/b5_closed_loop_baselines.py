from __future__ import annotations

from typing import Any

import numpy as np

from .b5_closed_loop_metrics import mean_or_zero
from .b5_epistemic_pragmatic_values import compute_closed_loop_value, oracle_closed_loop_plan


def random_closed_loop_baseline(episode: dict[str, Any], config: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    region = int(rng.integers(0, grid_size * grid_size))
    action = {"action_type": "apply_local_damping", "region_id": region, "strength": 1.0}
    return baseline_output("random", bool(rng.integers(0, 2)), region, action)


def saliency_closed_loop_baseline(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    region = int(episode["ground_truth"].get("saliency_region", 0))
    action = {"action_type": "apply_local_damping", "region_id": region, "strength": 1.0}
    return baseline_output("saliency", False, -1, action)


def short_horizon_closed_loop_baseline(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    region = int(episode["ground_truth"].get("short_horizon_region", 0))
    action = {"action_type": "apply_local_damping", "region_id": region, "strength": 1.0}
    return baseline_output("short_horizon", False, -1, action)


def inspect_always_baseline(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gt = episode["ground_truth"]
    action = {"action_type": "do_nothing", "region_id": int(gt["oracle_inspect_region"]), "strength": 0.0}
    return baseline_output("inspect_always", True, int(gt["oracle_inspect_region"]), action)


def intervene_immediately_baseline(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    initial_region = int(episode["ground_truth"]["initial_trace_region"])
    action = dict(episode["ground_truth"]["oracle_intervention_action"])
    action["region_id"] = initial_region
    return baseline_output("intervene_immediately", False, -1, action)


def oracle_closed_loop_baseline(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    plan = oracle_closed_loop_plan(episode, config)
    action = dict(plan["intervention_action"])
    return baseline_output("oracle", bool(plan["inspect"]), int(plan["inspect_region"]), action)


def evaluate_b5_baselines(episodes: list[dict[str, Any]], config: dict[str, Any], seed: int = 0) -> tuple[dict[str, float], list[dict[str, Any]]]:
    scores: dict[str, list[float]] = {name: [] for name in ["random", "saliency", "short_horizon", "inspect_always", "intervene_immediately", "oracle"]}
    records = []
    for episode_id, episode in enumerate(episodes):
        outputs = [
            random_closed_loop_baseline(episode, config, seed + episode_id),
            saliency_closed_loop_baseline(episode, config),
            short_horizon_closed_loop_baseline(episode, config),
            inspect_always_baseline(episode, config),
            intervene_immediately_baseline(episode, config),
            oracle_closed_loop_baseline(episode, config),
        ]
        for output in outputs:
            score = compute_closed_loop_value(output, episode, config)
            name = output["baseline_name"]
            scores[name].append(score)
            records.append(
                {
                    "record_kind": "baseline",
                    "episode_id": episode_id,
                    "episode_type": episode["ground_truth"]["closed_loop_episode_type"],
                    "baseline_name": name,
                    "baseline_value": score,
                    "predicted_inspect_region": int(output["predicted_inspect_region"]),
                    "predicted_intervention_action_type": output["intervention_action"]["action_type"],
                    "predicted_intervention_region": int(output["intervention_action"]["region_id"]),
                    "gate_pass": int(score > 0.0),
                    "note": "closed-loop baseline",
                }
            )
    return {
        "random_closed_loop_score": mean_or_zero(scores["random"]),
        "saliency_closed_loop_score": mean_or_zero(scores["saliency"]),
        "short_horizon_closed_loop_score": mean_or_zero(scores["short_horizon"]),
        "inspect_always_score": mean_or_zero(scores["inspect_always"]),
        "intervene_immediately_score": mean_or_zero(scores["intervene_immediately"]),
        "oracle_closed_loop_score": mean_or_zero(scores["oracle"]),
    }, records


def baseline_output(name: str, inspect_chosen: bool, inspect_region: int, action: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_name": name,
        "inspect_chosen": inspect_chosen,
        "predicted_inspect_region": int(inspect_region),
        "intervention_action": action,
    }
