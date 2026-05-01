from __future__ import annotations

from typing import Any

import numpy as np

from .b4_action_space import enumerate_candidate_actions


def compute_intervention_value(episode: dict[str, Any], action: dict[str, Any], config: dict[str, Any]) -> float:
    action_type = str(action.get("action_type", ""))
    if action_type in {"do_nothing", "inspect_only"}:
        return 0.0
    gt = episode["ground_truth"]
    key = action_key(action)
    if key in gt.get("intervention_values", {}):
        return float(gt["intervention_values"][key])
    oracle = gt["oracle_best_action"]
    if int(action.get("region_id", -1)) == int(oracle["region_id"]):
        return 0.35
    if action_type == str(oracle["action_type"]):
        return 0.10
    return 0.0


def compute_family_intervention_values(
    episode: dict[str, Any],
    family: str,
    config: dict[str, Any],
) -> dict[tuple[str, int], float]:
    family_key = canonical_family(family)
    expected = episode["ground_truth"]["family_best_actions"][family_key]
    values: dict[tuple[str, int], float] = {}
    for action in enumerate_candidate_actions(config):
        value = 0.0
        if str(action["action_type"]) not in {"do_nothing", "inspect_only"}:
            same_region = int(action["region_id"]) == int(expected["region_id"])
            same_type = str(action["action_type"]) == str(expected["action_type"])
            if same_region and same_type:
                value = 1.0
            elif same_region:
                value = 0.35
            elif same_type:
                value = 0.10
        values[(str(action["action_type"]), int(action["region_id"]))] = value
    return values


def compute_oracle_best_intervention(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    best_action = None
    best_value = -np.inf
    for action in enumerate_candidate_actions(config):
        value = compute_intervention_value(episode, action, config)
        if value > best_value:
            best_action = action
            best_value = value
    return dict(best_action or {"action_type": "do_nothing", "region_id": 0, "strength": 0.0})


def compute_wrong_region_penalty(
    episode: dict[str, Any],
    correct_action: dict[str, Any],
    wrong_region_action: dict[str, Any],
    config: dict[str, Any],
) -> float:
    return max(0.0, compute_intervention_value(episode, correct_action, config) - compute_intervention_value(episode, wrong_region_action, config))


def wrong_region_action_for(episode: dict[str, Any], correct_action: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    correct = int(correct_action["region_id"])
    saliency = int(episode["ground_truth"].get("saliency_region", -1))
    wrong = saliency if saliency != correct and 0 <= saliency < grid_size * grid_size else (correct + grid_size + 1) % (grid_size * grid_size)
    return {
        "action_type": str(correct_action["action_type"]),
        "region_id": int(wrong),
        "strength": float(correct_action.get("strength", 1.0)),
    }


def action_key(action: dict[str, Any]) -> str:
    return f"{action.get('action_type')}:{int(action.get('region_id', -1))}"


def canonical_family(family: str) -> str:
    value = str(family)
    if value in {"recurrent", "recurrent_goal"}:
        return "recurrent"
    if value in {"field", "field_goal"}:
        return "field"
    if value in {"schema", "schema_goal"}:
        return "schema"
    return "recurrent"

