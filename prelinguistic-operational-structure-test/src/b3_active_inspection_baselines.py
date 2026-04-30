from __future__ import annotations

from typing import Any

import numpy as np

from .b3_information_gain import compute_information_gain_after_inspection
from .features import region_logits_from_map
from .models.flow_checkpoint_model import FlowCheckpointModel
from .model_io import make_model_batch
from .inspect_policy import select_region_from_logits


def random_inspection_policy(episode: dict[str, Any], config: dict[str, Any], seed: int) -> dict[str, Any]:
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    rng = np.random.default_rng(seed)
    return {"inspect_region": int(rng.integers(0, grid_size * grid_size)), "policy_source": "random_inspection_baseline"}


def saliency_inspection_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    saliency = int(episode["ground_truth"].get("saliency_region", episode["ground_truth"].get("early_saliency_region", 0)))
    return {"inspect_region": saliency, "policy_source": "saliency_inspection_baseline"}


def short_horizon_checkpoint_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    saliency = int(episode["ground_truth"].get("saliency_region", episode["ground_truth"].get("early_saliency_region", 0)))
    return {"inspect_region": saliency, "policy_source": "short_horizon_checkpoint_baseline"}


def oracle_inspection_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {"inspect_region": int(episode["ground_truth"]["oracle_best_inspect_region"]), "policy_source": "oracle_inspection_baseline"}


def evaluate_b3_baselines(episodes: list[dict[str, Any]], config: dict[str, Any], seed: int = 0) -> tuple[dict[str, float], list[dict[str, Any]]]:
    names = ["random", "saliency", "short_horizon", "oracle"]
    scores = {name: [] for name in names}
    gains = {name: [] for name in names}
    records = []
    for idx, episode in enumerate(episodes):
        policies = {
            "random": random_inspection_policy(episode, config, seed + idx * 101),
            "saliency": saliency_inspection_policy(episode, config),
            "short_horizon": short_horizon_checkpoint_policy(episode, config),
            "oracle": oracle_inspection_policy(episode, config),
        }
        oracle_region = int(episode["ground_truth"]["oracle_best_inspect_region"])
        for name, policy in policies.items():
            region = int(policy["inspect_region"])
            score = 1.0 if region == oracle_region else 0.0
            gain = compute_information_gain_after_inspection(None, episode, region, config)["absolute_gain"]
            scores[name].append(score)
            gains[name].append(gain)
            records.append(
                {
                    "baseline": name,
                    "episode_id": idx,
                    "inspect_region": region,
                    "oracle_best_inspect_region": oracle_region,
                    "score": score,
                    "gain": gain,
                }
            )
    return {
        "random_inspection_score": mean_or_zero(scores["random"]),
        "saliency_inspection_score": mean_or_zero(scores["saliency"]),
        "short_horizon_inspection_score": mean_or_zero(scores["short_horizon"]),
        "oracle_inspection_score": mean_or_zero(scores["oracle"]),
        "random_information_gain": mean_or_zero(gains["random"]),
        "saliency_information_gain": mean_or_zero(gains["saliency"]),
        "short_horizon_information_gain": mean_or_zero(gains["short_horizon"]),
        "oracle_information_gain": mean_or_zero(gains["oracle"]),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
