from __future__ import annotations

from typing import Any


def delayed_indirect_success(action: dict[str, Any] | None, evaluator: dict[str, Any]) -> bool:
    if not action or action.get("action_type") != "indirect_stabilize":
        return False
    backfire = float(evaluator.get("backfire_probability", 0.0))
    delay = int(evaluator.get("delay_steps", 0))
    return backfire <= 0.45 and delay <= 5


def credit_assignment_accuracy(action: dict[str, Any] | None, evaluator: dict[str, Any]) -> float:
    if int(evaluator.get("delay_steps", 0)) <= 0:
        return 1.0
    return 1.0 if delayed_indirect_success(action, evaluator) else 0.0


def premature_direct_action(action: dict[str, Any] | None, evaluator: dict[str, Any]) -> bool:
    if int(evaluator.get("delay_steps", 0)) <= 0:
        return False
    return bool(action and action.get("action_type") == "apply_local_damping")


def backfire_avoidance(action: dict[str, Any] | None, evaluator: dict[str, Any]) -> float:
    backfire = float(evaluator.get("backfire_probability", 0.0))
    if backfire <= 0.45:
        return 1.0 if action and action.get("action_type") == "indirect_stabilize" else 0.0
    return 1.0 if action is None or action.get("action_type") == "abstain" else 0.0

