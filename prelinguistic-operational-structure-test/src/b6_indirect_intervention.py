from __future__ import annotations

from typing import Any


def find_indirect_intervention_candidates(episode_bundle: dict[str, Any], target_region: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    mask = episode_bundle["model_input"]["actionability_mask"]
    target_info = mask[int(target_region)]
    indirect = target_info.get("indirect_target_region")
    if indirect is None:
        return []
    info = mask[int(indirect)]
    if not bool(info.get("intervenable", False)) or bool(info.get("unsafe", False)) or bool(info.get("irreversible", False)):
        return []
    return [{"action_type": "indirect_stabilize", "region_id": int(indirect), "strength": 1.0}]


def evaluate_indirect_intervention_choice(policy_output: dict[str, Any], evaluator_ground_truth: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    expected = evaluator_ground_truth["expected_decision"]
    if expected.get("mode") != "indirect":
        return {"indirect_intervention_accuracy": 1.0}
    action = policy_output["intervention_decision"].get("action")
    expected_action = expected.get("action")
    hit = action is not None and expected_action is not None and action["action_type"] == expected_action["action_type"] and int(action["region_id"]) == int(expected_action["region_id"])
    return {"indirect_intervention_accuracy": 1.0 if hit else 0.0}


def apply_indirect_intervention(episode_bundle: dict[str, Any], indirect_action: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    candidates = find_indirect_intervention_candidates(episode_bundle, int(episode_bundle["metadata"]["target_region"]), config)
    valid = any(candidate["action_type"] == indirect_action["action_type"] and int(candidate["region_id"]) == int(indirect_action["region_id"]) for candidate in candidates)
    return {"indirect_value": 1.0 if valid else 0.0, "applied": float(valid)}
