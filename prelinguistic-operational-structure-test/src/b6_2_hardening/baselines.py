from __future__ import annotations

from typing import Any, Callable

from .policy import choose_indirect, estimate_risk


BASELINES = [
    "risk_blind",
    "mask_only",
    "random",
    "always_abstain",
    "oracle",
    "trace_only",
    "state_only",
    "conservative_uncertainty",
]


def risk_blind(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return output("risk_blind", {"action_type": "apply_local_damping", "region_id": int(episode["model_input"]["visible_state"]["state_target_hint"])}, False)


def mask_only(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_input = episode["model_input"]
    mask = model_input.get("actionability_mask")
    target = int(model_input["visible_state"]["state_target_hint"])
    if not mask:
        return output("mask_only", None, False, "missing_mask")
    info = mask[target]
    if info.get("unsafe") or info.get("irreversible"):
        action = choose_indirect(model_input, target, config)
        return output("mask_only", action, False, "mask_blocks" if action is None else "")
    return output("mask_only", {"action_type": "apply_local_damping", "region_id": target}, False)


def random_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    seed = int(episode["model_input"]["episode_id"])
    target = int(episode["model_input"]["visible_state"]["state_target_hint"])
    choice = seed % 4
    action = None if choice == 0 else {"action_type": "apply_local_damping" if choice < 3 else "indirect_stabilize", "region_id": target}
    return output("random", action, choice == 3, "random" if action is None else "")


def always_abstain(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return output("always_abstain", None, False, "always_abstain")


def oracle(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    out = output("oracle", episode["oracle_baseline_view"]["expected_action"], bool(episode["oracle_baseline_view"]["expected_inspect"]))
    action = out["action"]
    if action and action.get("action_type") == "indirect_stabilize":
        out["delayed_credit"] = {"credit_assigned": True, "no_effect": False, "backfire": False, "outcome_type": "success"}
    return out


def trace_only(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trace_region = episode["model_input"].get("previous_trace_state", {}).get("region")
    if trace_region is None:
        return output("trace_only", None, True, "missing_trace")
    return output("trace_only", {"action_type": "apply_local_damping", "region_id": int(trace_region)}, False)


def state_only(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    target = int(episode["model_input"]["visible_state"]["state_target_hint"])
    risk = float(episode["model_input"]["visible_state"]["risk_history_score"])
    if risk >= 0.72:
        return output("state_only", None, False, "state_risk")
    return output("state_only", {"action_type": "apply_local_damping", "region_id": target}, False)


def conservative_uncertainty(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trace_confidence = float(episode["model_input"].get("previous_trace_state", {}).get("confidence", 0.0))
    risk = estimate_risk(episode["model_input"], int(episode["model_input"]["visible_state"]["state_target_hint"]))
    if trace_confidence < 0.70 or risk >= 0.60:
        return output("conservative_uncertainty", None, True, "uncertain")
    return state_only(episode, config)


def output(name: str, action: dict[str, Any] | None, inspect: bool, reason: str = "") -> dict[str, Any]:
    return {"policy_name": name, "action": action, "inspect": inspect, "abstained": action is None, "failure_reason": reason}


def get_baseline(name: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    return {
        "risk_blind": risk_blind,
        "mask_only": mask_only,
        "random": random_policy,
        "always_abstain": always_abstain,
        "oracle": oracle,
        "trace_only": trace_only,
        "state_only": state_only,
        "conservative_uncertainty": conservative_uncertainty,
    }[name]
