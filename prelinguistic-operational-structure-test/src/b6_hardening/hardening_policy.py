from __future__ import annotations

from typing import Any

from .inspect_cost import should_inspect
from .risk_reward_conflict import choose_risk_reward_action


def hardening_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_input = episode["model_input"]
    visible = model_input["visible_state"]
    mask = model_input.get("actionability_mask")
    target = int(visible["target_region"])
    indirect = int(visible["indirect_target_region"])
    uncertainty = float(visible.get("mask_uncertainty", 0.0))
    risk_signal = float(visible.get("history_risk_signal", 0.0))
    latent = str(visible.get("latent_risk_marker", "safe"))
    inspect_cost = float(model_input.get("cost_config_public", {}).get("inspection_cost", 0.0))
    inspect = should_inspect(model_input, inspect_cost)

    mask_info = None if mask is None else mask.get(target)
    mask_says_unsafe = bool(mask_info and mask_info.get("unsafe", False))
    mask_says_irreversible = bool(mask_info and mask_info.get("irreversible", False))
    mask_says_blocked = bool(mask_info and not mask_info.get("directly_intervenable", True))
    estimated_irreversible = latent == "irreversible" or mask_says_irreversible
    estimated_unsafe = risk_signal >= 0.70 or mask_says_unsafe

    if estimated_irreversible:
        action = {"action_type": "abstain", "region_id": None}
        reason = "irreversible_risk"
    elif estimated_unsafe or mask_says_blocked:
        if visible.get("delayed_indirect_hint") and uncertainty <= 0.35:
            action = {"action_type": "indirect_stabilize", "region_id": indirect}
            reason = None
        elif visible.get("delayed_indirect_hint") and risk_signal < 0.95:
            action = {"action_type": "indirect_stabilize", "region_id": indirect}
            reason = None
        else:
            action = {"action_type": "abstain", "region_id": None}
            reason = "uncertain_or_unsafe"
    else:
        action = choose_risk_reward_action(risk_signal, visible.get("reward_estimate", {}))
        if action["region_id"] == "target":
            action["region_id"] = target
        elif action["region_id"] == "indirect":
            action["region_id"] = indirect
        reason = "risk_reward_guard" if action["action_type"] == "abstain" else None

    if action["action_type"] == "abstain":
        action = None
    return policy_output("hardening_policy", action, inspect, reason, mask is not None)


def policy_output(policy_name: str, action: dict[str, Any] | None, inspect: bool, reason: str | None, mask_used: bool) -> dict[str, Any]:
    return {
        "policy_name": policy_name,
        "inspect": bool(inspect),
        "action": action,
        "abstained": action is None,
        "abstain_reason": reason or ("abstain" if action is None else ""),
        "provenance": {
            "actionability_mask_used": bool(mask_used),
            "oracle_value_used": False,
            "oracle_risk_value_used": False,
        },
    }
