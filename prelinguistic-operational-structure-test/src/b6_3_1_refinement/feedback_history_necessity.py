from __future__ import annotations

from typing import Any


FEEDBACK_HISTORY_CONDITIONS = [
    "feedback_required_trace_repair",
    "history_required_delayed_credit",
    "history_required_indirect_discovery",
    "feedback_required_risk_update",
]


def apply_feedback_history_condition(episode: dict[str, Any], condition: str) -> None:
    visible = episode["model_input"]["visible_state"]
    evaluator = episode["evaluator_ground_truth"]
    target = int(evaluator["target_region"])
    wrong = int(evaluator["wrong_target_region"])
    indirect = int(evaluator["indirect_target_region"])
    visible["public_state_available"] = False
    visible["state_target_hint"] = wrong
    visible["state_target_confidence"] = 0.0
    episode["model_input"]["previous_trace_state"] = {"region": wrong, "confidence": 0.88, "candidate_regions": [wrong, target], "trace_mode": "wrong"}
    evaluator["expected_action"] = {"action_type": "apply_local_damping", "region_id": target}
    evaluator["expected_inspect"] = False

    if condition == "feedback_required_trace_repair":
        visible["history_supported_region"] = None
        visible["history_confidence"] = 0.0
        visible["feedback_history_region"] = target
        visible["feedback_confidence"] = 0.92
        evaluator["required_repair_source"] = "feedback"
    elif condition == "history_required_delayed_credit":
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.90
        visible["feedback_history_region"] = None
        visible["feedback_confidence"] = 0.0
        visible["pending_indirect_actions"] = [{"region_id": indirect, "expected_effect_signature": "stabilize_target", "delay_steps": 5}]
        visible["outcome_history"] = [{"after_steps": 5, "effect_signature": "stabilize_target", "backfire": False, "no_effect": False}]
        visible["delay_steps"] = 5
        evaluator["expected_action"] = {"action_type": "indirect_stabilize", "region_id": indirect}
        evaluator["required_repair_source"] = "history"
    elif condition == "history_required_indirect_discovery":
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.84
        visible["feedback_history_region"] = None
        visible["feedback_confidence"] = 0.0
        visible["candidate_indirect_regions"] = []
        visible["indirect_history_paths"] = [{"region_id": indirect, "causal_strength": 0.90, "backfire_estimate": 0.10}]
        visible["risk_history_score"] = 0.85
        evaluator["expected_action"] = {"action_type": "indirect_stabilize", "region_id": indirect}
        evaluator["required_repair_source"] = "history"
    elif condition == "feedback_required_risk_update":
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.65
        visible["feedback_history_region"] = target
        visible["feedback_confidence"] = 0.88
        visible["risk_history_score"] = 0.10
        visible["feedback_risk_marker"] = "unsafe"
        evaluator["expected_action"] = None
        evaluator["required_repair_source"] = "feedback"

