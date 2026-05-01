from __future__ import annotations

from typing import Any

import numpy as np

from .b4_intervention_values import compute_intervention_value, wrong_region_action_for


def construct_wrong_action_variants(
    episode: dict[str, Any],
    correct_action: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    wrong_type = first_wrong_action_type(str(correct_action["action_type"]))
    wrong_region = wrong_region_action_for(episode, correct_action, config)["region_id"]
    strength = float(correct_action.get("strength", 1.0))
    return {
        "correct_action": dict(correct_action),
        "wrong_action": {"action_type": wrong_type, "region_id": int(correct_action["region_id"]), "strength": strength},
        "wrong_region": {"action_type": str(correct_action["action_type"]), "region_id": int(wrong_region), "strength": strength},
        "wrong_action_wrong_region": {"action_type": wrong_type, "region_id": int(wrong_region), "strength": strength},
    }


def evaluate_wrong_action_wrong_region_stress(
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    wrong_action_penalties = []
    wrong_region_penalties = []
    wrong_both_penalties = []
    records = []
    for episode_id, episode in enumerate(episodes):
        correct = episode["ground_truth"]["oracle_best_action"]
        variants = construct_wrong_action_variants(episode, correct, config)
        correct_value = compute_intervention_value(episode, variants["correct_action"], config)
        wrong_action_value = compute_intervention_value(episode, variants["wrong_action"], config)
        wrong_region_value = compute_intervention_value(episode, variants["wrong_region"], config)
        wrong_both_value = compute_intervention_value(episode, variants["wrong_action_wrong_region"], config)
        wrong_action_penalty = max(0.0, correct_value - wrong_action_value)
        wrong_region_penalty = max(0.0, correct_value - wrong_region_value)
        wrong_both_penalty = max(0.0, correct_value - wrong_both_value)
        wrong_action_penalties.append(wrong_action_penalty)
        wrong_region_penalties.append(wrong_region_penalty)
        wrong_both_penalties.append(wrong_both_penalty)
        records.append(
            {
                "record_kind": "wrong_action_stress",
                "seed": seed,
                "episode_id": episode_id,
                "audit_type": "wrong_action_wrong_region_stress",
                "correct_action_value": correct_value,
                "wrong_action_value": wrong_action_value,
                "wrong_region_value": wrong_region_value,
                "wrong_action_wrong_region_value": wrong_both_value,
                "wrong_action_penalty": wrong_action_penalty,
                "wrong_region_penalty": wrong_region_penalty,
                "wrong_action_wrong_region_penalty": wrong_both_penalty,
                "gate_pass": int(wrong_action_penalty >= 0.0 and wrong_region_penalty >= 0.0),
                "note": "wrong action and wrong region intervention stress",
            }
        )
    return {
        "wrong_action_penalty": mean_or_zero(wrong_action_penalties),
        "wrong_region_penalty": mean_or_zero(wrong_region_penalties),
        "wrong_action_wrong_region_penalty": mean_or_zero(wrong_both_penalties),
    }, records


def first_wrong_action_type(action_type: str) -> str:
    candidates = ["apply_local_damping", "apply_local_push", "block_force_region", "stabilize_trace_region"]
    for candidate in candidates:
        if candidate != action_type:
            return candidate
    return "apply_local_damping"


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

