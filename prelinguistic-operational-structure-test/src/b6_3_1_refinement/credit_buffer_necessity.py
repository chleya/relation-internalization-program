from __future__ import annotations

from typing import Any


CREDIT_BUFFER_CONDITIONS = [
    "delay5_credit_buffer_required",
    "delay5_multiple_pending_actions",
    "delay5_backfire_after_success",
    "delay5_no_effect_vs_delayed_success",
]


def apply_credit_buffer_condition(episode: dict[str, Any], condition: str) -> None:
    visible = episode["model_input"]["visible_state"]
    evaluator = episode["evaluator_ground_truth"]
    target = int(evaluator["target_region"])
    indirect = int(evaluator["indirect_target_region"])
    decoy = (indirect + 7) % 64
    visible["public_state_available"] = False
    visible["state_target_hint"] = int(evaluator["state_decoy_region"])
    visible["state_target_confidence"] = 0.0
    visible["history_supported_region"] = target
    visible["history_confidence"] = 0.74
    visible["risk_history_score"] = 0.82
    visible["delay_steps"] = 5
    visible["expected_delayed_effect_signature"] = "stabilize_target"
    visible["candidate_indirect_regions"] = [
        {"region_id": decoy, "causal_strength": 0.86, "backfire_estimate": 0.12},
        {"region_id": indirect, "causal_strength": 0.82, "backfire_estimate": 0.10},
    ]
    evaluator["expected_action"] = {"action_type": "indirect_stabilize", "region_id": indirect}
    evaluator["expected_inspect"] = False
    evaluator["required_credit_buffer"] = True

    if condition == "delay5_credit_buffer_required":
        visible["pending_indirect_actions"] = [{"region_id": indirect, "expected_effect_signature": "stabilize_target", "delay_steps": 5}]
        visible["outcome_history"] = [{"after_steps": 5, "effect_signature": "stabilize_target", "backfire": False, "no_effect": False}]
    elif condition == "delay5_multiple_pending_actions":
        visible["pending_indirect_actions"] = [
            {"region_id": decoy, "expected_effect_signature": "decoy_effect", "delay_steps": 5},
            {"region_id": indirect, "expected_effect_signature": "stabilize_target", "delay_steps": 5},
        ]
        visible["outcome_history"] = [{"after_steps": 5, "effect_signature": "stabilize_target", "backfire": False, "no_effect": False}]
    elif condition == "delay5_backfire_after_success":
        visible["pending_indirect_actions"] = [{"region_id": indirect, "expected_effect_signature": "stabilize_target", "delay_steps": 5}]
        visible["outcome_history"] = [
            {"after_steps": 5, "effect_signature": "stabilize_target", "backfire": False, "no_effect": False},
            {"after_steps": 6, "effect_signature": "destabilize_target", "backfire": True, "no_effect": False},
        ]
        evaluator["expected_action"] = None
    elif condition == "delay5_no_effect_vs_delayed_success":
        visible["pending_indirect_actions"] = [
            {"region_id": decoy, "expected_effect_signature": "stabilize_target", "delay_steps": 5},
            {"region_id": indirect, "expected_effect_signature": "stabilize_target", "delay_steps": 5},
        ]
        visible["outcome_history"] = [
            {"after_steps": 5, "region_id": decoy, "effect_signature": "stabilize_target", "backfire": False, "no_effect": True},
            {"after_steps": 5, "region_id": indirect, "effect_signature": "stabilize_target", "backfire": False, "no_effect": False},
        ]


def assign_credit_with_buffer(model_input: dict[str, Any], disabled: set[str] | None = None) -> dict[str, Any]:
    disabled = disabled or set()
    visible = model_input.get("visible_state", {})
    if "credit_buffer" in disabled:
        return {"action": None, "credit": {"credit_assigned": False, "no_effect": False, "backfire": False, "outcome_type": "buffer_disabled"}}

    pending = list(visible.get("pending_indirect_actions", []))
    outcomes = list(visible.get("outcome_history", []))
    for action in pending:
        region = int(action["region_id"])
        for outcome in outcomes:
            if int(outcome.get("after_steps", -1)) != int(action.get("delay_steps", 5)):
                continue
            if "region_id" in outcome and int(outcome["region_id"]) != region:
                continue
            if outcome.get("backfire"):
                return {"action": None, "credit": {"credit_assigned": False, "no_effect": False, "backfire": True, "outcome_type": "backfire"}}
            if outcome.get("no_effect"):
                continue
            if outcome.get("effect_signature") == action.get("expected_effect_signature"):
                return {
                    "action": {"action_type": "indirect_stabilize", "region_id": region},
                    "credit": {"credit_assigned": True, "no_effect": False, "backfire": False, "outcome_type": "success"},
                }
    return {"action": None, "credit": {"credit_assigned": False, "no_effect": True, "backfire": False, "outcome_type": "no_effect"}}

