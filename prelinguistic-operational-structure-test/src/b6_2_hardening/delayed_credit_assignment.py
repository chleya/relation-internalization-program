from __future__ import annotations

from typing import Any


def register_pending_indirect_action(action: dict[str, Any], model_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    visible = model_input.get("visible_state", {})
    return {
        "action_type": action.get("action_type"),
        "region_id": int(action.get("region_id")),
        "expected_effect_signature": visible.get("expected_delayed_effect_signature", "stabilize_target"),
        "delay_steps": int(visible.get("delay_steps", 0)),
    }


def update_delayed_credit(pending_action: dict[str, Any], outcome_history: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    expected = pending_action.get("expected_effect_signature")
    delay_steps = int(pending_action.get("delay_steps", 0))
    for outcome in outcome_history:
        if int(outcome.get("after_steps", -1)) != delay_steps:
            continue
        signature = outcome.get("effect_signature")
        if outcome.get("backfire"):
            return {"outcome_type": "backfire", "matched": False, "observed_signature": signature}
        if outcome.get("no_effect"):
            return {"outcome_type": "no_effect", "matched": False, "observed_signature": signature}
        return {"outcome_type": "success", "matched": signature == expected, "observed_signature": signature}
    return {"outcome_type": "pending", "matched": False, "observed_signature": None}


def assign_credit_to_indirect_action(pending_action: dict[str, Any], outcome_history: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    update = update_delayed_credit(pending_action, outcome_history, config)
    credit_assigned = bool(update["outcome_type"] == "success" and update["matched"])
    return {
        **update,
        "credit_assigned": credit_assigned,
        "no_effect": update["outcome_type"] == "no_effect",
        "backfire": update["outcome_type"] == "backfire",
    }


def select_delayed_indirect_candidate(model_input: dict[str, Any], target: int, config: dict[str, Any]) -> dict[str, Any] | None:
    mask = model_input.get("actionability_mask")
    if mask and target in mask and mask[target].get("indirect_target_region") is not None:
        region = int(mask[target]["indirect_target_region"])
        return {"action_type": "indirect_stabilize", "region_id": region}

    candidates = list(model_input.get("visible_state", {}).get("candidate_indirect_regions", []))
    viable = [row for row in candidates if float(row.get("backfire_estimate", 1.0)) <= 0.45]
    if not viable:
        return None
    best = max(viable, key=lambda row: float(row.get("causal_strength", 0.0)) - float(row.get("backfire_estimate", 0.0)))
    if float(best.get("causal_strength", 0.0)) < 0.50:
        return None
    return {"action_type": "indirect_stabilize", "region_id": int(best["region_id"])}


def delayed_indirect_policy(model_input: dict[str, Any], target: int, config: dict[str, Any]) -> dict[str, Any]:
    action = select_delayed_indirect_candidate(model_input, target, config)
    if action is None:
        return {
            "action": None,
            "pending_action": None,
            "credit": {"credit_assigned": False, "no_effect": False, "backfire": False, "outcome_type": "no_action"},
        }
    pending = register_pending_indirect_action(action, model_input, config)
    credit = assign_credit_to_indirect_action(pending, list(model_input.get("visible_state", {}).get("outcome_history", [])), config)
    return {"action": action, "pending_action": pending, "credit": credit}
