from __future__ import annotations

import inspect
from typing import Any, Callable


POLICY_NAMES = [
    "b64_policy_reference",
    "b64_1_transfer_policy",
    "state_only",
    "mask_only",
    "trace_only",
    "random",
    "always_abstain",
    "conservative_uncertainty",
    "oracle",
]


def b641_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_input = episode["model_input"]
    visible = model_input.get("visible_state", {})
    action = None
    source = "none"
    delayed = assign_credit(model_input)
    hidden = discover_indirect(model_input)
    if delayed is not None:
        action = delayed
        source = "delayed_credit"
    elif hidden is not None:
        action = hidden
        source = "hidden_indirect"
    elif visible.get("feedback_history_region") is not None and float(visible.get("feedback_confidence", 0.0)) >= 0.66:
        action = {"action_type": "apply_local_damping", "region_id": int(visible["feedback_history_region"])}
        source = "feedback"
    elif visible.get("transition_consistency_region") is not None and float(visible.get("transition_confidence", 0.0)) >= 0.75:
        action = {"action_type": "apply_local_damping", "region_id": int(visible["transition_consistency_region"])}
        source = "transition_history"
    elif visible.get("history_supported_region") is not None and float(visible.get("history_confidence", 0.0)) >= 0.72:
        action = {"action_type": "apply_local_damping", "region_id": int(visible["history_supported_region"])}
        source = "history"
    else:
        source = "abstain_uncertain"
    return {
        "policy_name": "b64_1_transfer_policy",
        "inspect": False,
        "action": action,
        "transfer_source": source,
        "delayed_credit": {"credit_assigned": delayed is not None, "buffer_used": delayed is not None},
        "hidden_indirect": {"success": hidden is not None, "candidate_search_used": False},
        "provenance": {"policy_uses_model_input_only": True, "oracle_value_used": False},
    }


def assign_credit(model_input: dict[str, Any]) -> dict[str, Any] | None:
    visible = model_input.get("visible_state", {})
    for pending in visible.get("pending_indirect_actions", []):
        for outcome in visible.get("outcome_history", []):
            if outcome.get("source_action_id") == pending.get("action_id") and outcome.get("signature") == pending.get("expected_signature"):
                return {"action_type": "indirect_stabilize", "region_id": int(pending["region_id"])}
    return None


def discover_indirect(model_input: dict[str, Any]) -> dict[str, Any] | None:
    visible = model_input.get("visible_state", {})
    for item in visible.get("exploration_history", []):
        if item.get("success") and item.get("effect_signature") == "stabilize_target":
            return {"action_type": "indirect_stabilize", "region_id": int(item["region_id"])}
    return None


def baseline_policy(name: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    def output(action: dict[str, Any] | None, source: str) -> dict[str, Any]:
        return {
            "policy_name": name,
            "inspect": False,
            "action": action,
            "transfer_source": source,
            "delayed_credit": {"credit_assigned": False, "buffer_used": False},
            "hidden_indirect": {"success": False, "candidate_search_used": source == "mask"},
            "provenance": {"policy_uses_model_input_only": True},
        }

    def run(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        model_input = episode["model_input"]
        visible = model_input.get("visible_state", {})
        trace = model_input.get("previous_trace_state", {})
        if name == "b64_policy_reference":
            if visible.get("public_state_available", True) and float(visible.get("state_target_confidence", 0.0)) >= 0.55:
                return output({"action_type": "apply_local_damping", "region_id": int(visible["state_target_hint"])}, "first_pass_state")
            if trace.get("region") is not None and float(trace.get("confidence", 0.0)) >= 0.55:
                return output({"action_type": "apply_local_damping", "region_id": int(trace["region"])}, "first_pass_trace")
            return output(None, "first_pass_abstain")
        if name == "state_only":
            if not visible.get("public_state_available", True) or float(visible.get("state_target_confidence", 0.0)) < 0.55:
                return output(None, "state_unavailable")
            return output({"action_type": "apply_local_damping", "region_id": int(visible["state_target_hint"])}, "state")
        if name == "mask_only":
            for info in (model_input.get("actionability_mask") or {}).values():
                if info.get("indirect_target_region") is not None:
                    return output({"action_type": "indirect_stabilize", "region_id": int(info["indirect_target_region"])}, "mask")
            return output(None, "mask_hidden")
        if name == "trace_only":
            if trace.get("region") is None:
                return output(None, "trace_missing")
            return output({"action_type": "apply_local_damping", "region_id": int(trace["region"])}, "trace")
        if name == "random":
            region = int(visible.get("state_target_hint", 0))
            return output(None if int(model_input["episode_id"]) % 3 == 0 else {"action_type": "apply_local_damping", "region_id": region}, "random")
        if name == "always_abstain":
            return output(None, "always_abstain")
        if name == "conservative_uncertainty":
            if float(trace.get("confidence", 0.0)) < 0.80 or not visible.get("public_state_available", True):
                return output(None, "conservative_abstain")
            return output({"action_type": "apply_local_damping", "region_id": int(trace["region"])}, "conservative_trace")
        if name == "oracle":
            expected = episode["oracle_baseline_view"]["expected_action"]
            out = output(expected, "oracle")
            if expected and expected.get("action_type") == "indirect_stabilize":
                out["delayed_credit"] = {"credit_assigned": True, "buffer_used": True}
                out["hidden_indirect"] = {"success": True, "candidate_search_used": False}
            return out
        raise KeyError(name)

    return run


def policy_for(name: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    if name == "b64_1_transfer_policy":
        return b641_policy
    return baseline_policy(name)


def audit_policy_integrity(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source = inspect.getsource(b641_policy) + inspect.getsource(assign_credit) + inspect.getsource(discover_indirect)
    forbidden = ["evaluator_ground_truth", "oracle_baseline_view", "expected_decision", "metadata"]
    forbidden_count = sum(1 for token in forbidden if token in source)
    original = b641_policy(episode, config)
    poisoned = {
        **episode,
        "evaluator_ground_truth": {**episode["evaluator_ground_truth"], "expected_action": None, "condition": "poisoned"},
        "metadata": {**episode["metadata"], "condition": "poisoned"},
    }
    changed = b641_policy(poisoned, config)
    return {
        "forbidden_reference_count": forbidden_count,
        "poisoned_evaluator_invariance_pass": comparable(original) == comparable(changed),
        "policy_uses_model_input_only": forbidden_count == 0,
    }


def comparable(output: dict[str, Any]) -> tuple[Any, ...]:
    action = output.get("action") or {}
    return action.get("action_type"), action.get("region_id"), output.get("transfer_source")
