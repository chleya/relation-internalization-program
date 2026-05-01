from __future__ import annotations

from typing import Any

import numpy as np

from .b42_action_type_env import allowed_action_types
from .b42_action_type_values import compute_action_type_value


def run_action_type_counterfactual(episode: dict[str, Any], selected_region: int, config: dict[str, Any]) -> dict[str, float]:
    return {
        action_type: compute_action_type_value(episode, action_type, int(selected_region), config)
        for action_type in allowed_action_types(config)
    }


def compute_action_type_counterfactual_sensitivity(counterfactual_results: dict[str, float], correct_action_type: str) -> float:
    correct = float(counterfactual_results.get(correct_action_type, 0.0))
    wrong_values = [float(value) for action_type, value in counterfactual_results.items() if action_type != correct_action_type]
    best_wrong = max(wrong_values) if wrong_values else 0.0
    return max(0.0, correct - best_wrong)


def evaluate_action_type_counterfactuals(
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    sensitivities = []
    records = []
    for episode_id, episode in enumerate(episodes):
        gt = episode["ground_truth"]
        correct = gt["oracle_best_action"]
        results = run_action_type_counterfactual(episode, int(correct["region_id"]), config)
        sensitivity = compute_action_type_counterfactual_sensitivity(results, str(correct["action_type"]))
        wrong_values = [float(value) for action_type, value in results.items() if action_type != correct["action_type"]]
        sensitivities.append(sensitivity)
        records.append(
            {
                "record_kind": "counterfactual",
                "seed": seed,
                "episode_id": episode_id,
                "family": gt.get("family", ""),
                "expected_region": int(correct["region_id"]),
                "expected_action_type": correct["action_type"],
                "correct_action_value": float(results[correct["action_type"]]),
                "wrong_action_value": max(wrong_values) if wrong_values else 0.0,
                "action_type_counterfactual_sensitivity": sensitivity,
                "gate_pass": int(sensitivity >= 0.0),
                "note": "fixed-region action-type counterfactual",
            }
        )
    return {"action_type_counterfactual_sensitivity": mean_or_zero(sensitivities)}, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

