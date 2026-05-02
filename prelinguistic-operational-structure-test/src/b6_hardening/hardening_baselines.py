from __future__ import annotations

from typing import Any, Callable

from .hardening_policy import policy_output


BASELINE_NAMES = [
    "risk_blind_policy",
    "mask_only_policy",
    "always_inspect_policy",
    "always_abstain_policy",
    "oracle_risk_policy",
    "random_policy",
]


def risk_blind_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    target = int(episode["model_input"]["visible_state"]["target_region"])
    return policy_output("risk_blind_policy", {"action_type": "apply_local_damping", "region_id": target}, False, None, False)


def mask_only_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_input = episode["model_input"]
    mask = model_input.get("actionability_mask")
    target = int(model_input["visible_state"]["target_region"])
    if mask is None:
        return policy_output("mask_only_policy", None, False, "missing_mask", False)
    info = mask[target]
    if bool(info.get("unsafe", False)) or bool(info.get("irreversible", False)) or not bool(info.get("intervenable", True)):
        return policy_output("mask_only_policy", None, False, "mask_blocks_action", True)
    return policy_output("mask_only_policy", {"action_type": "apply_local_damping", "region_id": target}, False, None, True)


def always_inspect_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    target = int(episode["model_input"]["visible_state"]["target_region"])
    return policy_output("always_inspect_policy", {"action_type": "apply_local_damping", "region_id": target}, True, None, bool(episode["model_input"].get("actionability_mask")))


def always_abstain_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return policy_output("always_abstain_policy", None, False, "always_abstain", bool(episode["model_input"].get("actionability_mask")))


def oracle_risk_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    oracle_action = episode["oracle_baseline_view"]["oracle_action"]
    target = int(episode["evaluator_ground_truth"]["target_region"])
    indirect = int(episode["evaluator_ground_truth"]["indirect_target_region"])
    if oracle_action["region_id"] == "target":
        action = {"action_type": oracle_action["action_type"], "region_id": target}
    elif oracle_action["region_id"] == "indirect":
        action = {"action_type": oracle_action["action_type"], "region_id": indirect}
    else:
        action = None
    return policy_output("oracle_risk_policy", action, bool(episode["oracle_baseline_view"].get("oracle_inspect", False)), None, True)


def random_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    seed = int(episode["metadata"]["stress_seed"])
    target = int(episode["model_input"]["visible_state"]["target_region"])
    choice = seed % 4
    if choice == 0:
        action = None
    elif choice == 1:
        action = {"action_type": "apply_local_damping", "region_id": target}
    else:
        action = {"action_type": "indirect_stabilize", "region_id": int(episode["model_input"]["visible_state"]["indirect_target_region"])}
    return policy_output("random_policy", action, choice == 3, "random_abstain" if action is None else None, False)


def get_policy(name: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    policies = {
        "risk_blind_policy": risk_blind_policy,
        "mask_only_policy": mask_only_policy,
        "always_inspect_policy": always_inspect_policy,
        "always_abstain_policy": always_abstain_policy,
        "oracle_risk_policy": oracle_risk_policy,
        "random_policy": random_policy,
    }
    return policies[name]

