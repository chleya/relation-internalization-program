from __future__ import annotations

from typing import Any

from .b5_closed_loop_env import b5_costs


def compute_epistemic_value(episode: dict[str, Any], inspect_region_id: int | None, config: dict[str, Any]) -> float:
    gt = episode["ground_truth"]
    if inspect_region_id is None or int(inspect_region_id) < 0:
        return 1.0 if not bool(gt["needs_inspection"]) else 0.0
    return 1.0 if int(inspect_region_id) == int(gt["oracle_inspect_region"]) else 0.0


def compute_pragmatic_value(episode: dict[str, Any], action: dict[str, Any], config: dict[str, Any]) -> float:
    gt = episode["ground_truth"]
    expected = gt["oracle_intervention_action"]
    if bool(gt.get("should_do_nothing", False)):
        return 1.0 if str(action.get("action_type")) == "do_nothing" else 0.0
    return 1.0 if same_action(action, expected) else 0.0


def compute_closed_loop_value(policy_output: dict[str, Any], episode: dict[str, Any], config: dict[str, Any]) -> float:
    gt = episode["ground_truth"]
    costs = b5_costs(config)
    inspect_region_id = policy_output.get("predicted_inspect_region")
    action = policy_output.get("intervention_action", {"action_type": "do_nothing", "region_id": -1})
    epistemic = compute_epistemic_value(episode, inspect_region_id, config)
    pragmatic = compute_pragmatic_value(episode, action, config)
    value = 0.5 * epistemic + 0.5 * pragmatic
    if policy_output.get("inspect_chosen", False) and not gt["needs_inspection"]:
        value -= costs["inspection_cost"]
    if not policy_output.get("inspect_chosen", False) and gt["needs_inspection"]:
        value -= costs["wrong_inspect_penalty"]
    if pragmatic <= 0.0 and not gt.get("should_do_nothing", False):
        value -= costs["wrong_intervention_penalty"]
    return max(0.0, min(1.0, float(value)))


def compute_wrong_inspect_penalty(episode: dict[str, Any], config: dict[str, Any]) -> float:
    correct = compute_epistemic_value(episode, int(episode["ground_truth"]["oracle_inspect_region"]), config)
    wrong = compute_epistemic_value(episode, int(episode["ground_truth"]["wrong_inspect_region"]), config)
    return max(0.0, correct - wrong)


def compute_wrong_intervention_penalty(episode: dict[str, Any], config: dict[str, Any]) -> float:
    correct = compute_pragmatic_value(episode, episode["ground_truth"]["oracle_intervention_action"], config)
    wrong_action = dict(episode["ground_truth"]["oracle_intervention_action"])
    wrong_action["region_id"] = int(episode["ground_truth"]["wrong_inspect_region"])
    wrong = compute_pragmatic_value(episode, wrong_action, config)
    return max(0.0, correct - wrong)


def oracle_closed_loop_plan(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return dict(episode["ground_truth"]["oracle_closed_loop_plan"])


def same_action(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return str(left.get("action_type")) == str(right.get("action_type")) and int(left.get("region_id", -1)) == int(right.get("region_id", -2))
