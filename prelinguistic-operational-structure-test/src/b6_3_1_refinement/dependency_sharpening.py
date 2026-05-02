from __future__ import annotations

from typing import Any


PUBLIC_STATE_DEPENDENCY_CONDITIONS = [
    "hide_public_state_cue",
    "hide_public_state_and_history",
    "hide_public_state_and_feedback",
    "hide_public_state_and_mask_target",
    "hide_public_state_and_candidate_search",
]


def apply_public_state_dependency_condition(episode: dict[str, Any], condition: str) -> None:
    visible = episode["model_input"]["visible_state"]
    evaluator = episode["evaluator_ground_truth"]
    target = int(evaluator["target_region"])
    decoy = int(evaluator["state_decoy_region"])
    visible["public_state_available"] = False
    visible["state_target_hint"] = decoy
    visible["state_target_confidence"] = 0.0
    visible["history_supported_region"] = target
    visible["history_confidence"] = 0.78
    visible["feedback_history_region"] = target
    visible["feedback_confidence"] = 0.74
    visible["inspection_observation_region"] = target
    visible["inspection_confidence"] = 0.70
    evaluator["expected_action"] = {"action_type": "apply_local_damping", "region_id": target}
    evaluator["expected_inspect"] = False
    evaluator["required_repair_source"] = "substitute"

    if condition == "hide_public_state_and_history":
        visible["history_supported_region"] = None
        visible["history_confidence"] = 0.0
        evaluator["required_repair_source"] = "feedback"
    elif condition == "hide_public_state_and_feedback":
        visible["feedback_history_region"] = None
        visible["feedback_confidence"] = 0.0
        evaluator["required_repair_source"] = "history"
    elif condition == "hide_public_state_and_mask_target":
        hide_mask_target(episode)
        evaluator["required_repair_source"] = "history_or_feedback"
    elif condition == "hide_public_state_and_candidate_search":
        visible["candidate_indirect_regions"] = []
        evaluator["required_repair_source"] = "history_or_feedback"


def hide_mask_target(episode: dict[str, Any]) -> None:
    target = int(episode["evaluator_ground_truth"]["target_region"])
    mask = episode["model_input"].get("actionability_mask") or {}
    if target in mask:
        mask[target].pop("unsafe", None)
        mask[target].pop("risk_cost", None)
        mask[target].pop("indirect_target_region", None)

