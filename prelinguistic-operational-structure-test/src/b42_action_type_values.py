from __future__ import annotations

from typing import Any

from .b42_action_type_env import action_key, allowed_action_types


def compute_action_type_value(episode: dict[str, Any], action_type: str, region_id: int, config: dict[str, Any]) -> float:
    values = episode["ground_truth"].get("action_type_values", {})
    key = action_key(str(action_type), int(region_id))
    if key in values:
        return float(values[key])
    correct = episode["ground_truth"]["oracle_best_action"]
    if int(region_id) == int(correct["region_id"]) and str(action_type) == str(correct["action_type"]):
        return 1.0
    if int(region_id) == int(correct["region_id"]):
        return 0.10
    if str(action_type) == str(correct["action_type"]):
        return 0.05
    return 0.0


def compute_action_type_value_table(episode: dict[str, Any], config: dict[str, Any]) -> dict[tuple[str, int], float]:
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    return {
        (action_type, region_id): compute_action_type_value(episode, action_type, region_id, config)
        for action_type in allowed_action_types(config)
        for region_id in range(grid_size * grid_size)
    }


def value_table_rows(episode: dict[str, Any], config: dict[str, Any], episode_id: int = 0) -> list[dict[str, Any]]:
    gt = episode["ground_truth"]
    correct = gt["oracle_best_action"]
    rows = []
    for (action_type, region_id), value in compute_action_type_value_table(episode, config).items():
        rows.append(
            {
                "episode_id": episode_id,
                "family": gt.get("family", ""),
                "action_type": action_type,
                "region_id": int(region_id),
                "value": float(value),
                "is_correct_region": int(int(region_id) == int(correct["region_id"])),
                "is_correct_action": int(action_type == correct["action_type"]),
            }
        )
    return rows


def compute_correct_region_wrong_action_penalty(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    correct = episode["ground_truth"]["oracle_best_action"]
    correct_value = compute_action_type_value(episode, correct["action_type"], int(correct["region_id"]), config)
    wrong_values = [
        compute_action_type_value(episode, wrong_action, int(correct["region_id"]), config)
        for wrong_action in episode["ground_truth"].get("wrong_action_types", [])
    ]
    wrong_action_value = max(wrong_values) if wrong_values else 0.0
    return {
        "correct_action_value": correct_value,
        "wrong_action_value": wrong_action_value,
        "correct_region_wrong_action_penalty": max(0.0, correct_value - wrong_action_value),
    }


def compute_family_action_value_map(episode: dict[str, Any], family: str, config: dict[str, Any]) -> dict[str, float]:
    region = int(episode["ground_truth"]["intervention_region"])
    return {
        action_type: compute_action_type_value(episode, action_type, region, config)
        for action_type in allowed_action_types(config)
    }

