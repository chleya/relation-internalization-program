from __future__ import annotations

import inspect
from typing import Any, Callable


def b64_transfer_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_input = episode["model_input"]
    visible = model_input.get("visible_state", {})
    trace = model_input.get("previous_trace_state", {})
    action = None
    source = "none"
    inspect_requested = False

    delayed = assign_delayed_credit(model_input)
    hidden = discover_indirect_from_history(model_input)
    if delayed["action"] is not None:
        action = delayed["action"]
        source = "delayed_credit"
    elif hidden["action"] is not None:
        action = hidden["action"]
        source = "hidden_indirect_history"
    elif visible.get("feedback_history_region") is not None and float(visible.get("feedback_confidence", 0.0)) >= 0.66:
        action = {"action_type": "apply_local_damping", "region_id": int(visible["feedback_history_region"])}
        source = "feedback"
    elif visible.get("history_supported_region") is not None and float(visible.get("history_confidence", 0.0)) >= 0.65:
        action = {"action_type": "apply_local_damping", "region_id": int(visible["history_supported_region"])}
        source = "history"
    elif trace.get("region") is not None and float(trace.get("confidence", 0.0)) >= 0.70:
        action = {"action_type": "apply_local_damping", "region_id": int(trace["region"])}
        source = "trace"
    elif visible.get("public_state_available", True) and float(visible.get("state_target_confidence", 0.0)) >= 0.60:
        action = {"action_type": "apply_local_damping", "region_id": int(visible["state_target_hint"])}
        source = "state"
    else:
        inspect_requested = True
        source = "inspect_or_abstain"

    return {
        "policy_name": "b64_transfer_policy",
        "inspect": inspect_requested,
        "action": action,
        "transfer_source": source,
        "delayed_credit": {"credit_assigned": delayed["action"] is not None, "buffer_used": delayed["action"] is not None},
        "hidden_indirect": {"success": hidden["action"] is not None, "candidate_search_used": False, "discovery_source": hidden["source"]},
        "provenance": {"policy_uses_model_input_only": True, "oracle_value_used": False},
    }


def assign_delayed_credit(model_input: dict[str, Any]) -> dict[str, Any]:
    visible = model_input.get("visible_state", {})
    pending = visible.get("pending_indirect_actions", [])
    outcomes = visible.get("outcome_history", [])
    for action in pending:
        expected = action.get("expected_signature")
        for outcome in outcomes:
            if outcome.get("source_action_id") == action.get("action_id") and outcome.get("signature") == expected:
                return {"action": {"action_type": "indirect_stabilize", "region_id": int(action["region_id"])}}
    return {"action": None}


def discover_indirect_from_history(model_input: dict[str, Any]) -> dict[str, Any]:
    visible = model_input.get("visible_state", {})
    for row in visible.get("exploration_history", []):
        if row.get("success") and row.get("effect_signature") == "stabilize_target":
            return {"action": {"action_type": "indirect_stabilize", "region_id": int(row["region_id"])}, "source": "exploration_history"}
    for row in visible.get("indirect_history_paths", []):
        if row.get("success") and row.get("effect_signature") == "stabilize_target":
            return {"action": {"action_type": "indirect_stabilize", "region_id": int(row["region_id"])}, "source": "indirect_history"}
    return {"action": None, "source": "none"}


def baseline_policy(name: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    def output(action: dict[str, Any] | None, inspect_requested: bool = False, source: str = "") -> dict[str, Any]:
        return {
            "policy_name": name,
            "inspect": inspect_requested,
            "action": action,
            "transfer_source": source,
            "delayed_credit": {"credit_assigned": False, "buffer_used": False},
            "hidden_indirect": {"success": False, "candidate_search_used": source == "candidate_search", "discovery_source": source},
            "provenance": {"policy_uses_model_input_only": True},
        }

    def run(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        model_input = episode["model_input"]
        visible = model_input.get("visible_state", {})
        trace = model_input.get("previous_trace_state", {})
        if name == "b63_1_policy_reference":
            if visible.get("public_state_available", True) and float(visible.get("state_target_confidence", 0.0)) >= 0.55:
                return output({"action_type": "apply_local_damping", "region_id": int(visible["state_target_hint"])}, source="b63_reference_state")
            region = trace.get("region")
            return output(None if region is None else {"action_type": "apply_local_damping", "region_id": int(region)}, source="b63_reference_trace")
        if name == "state_only":
            if not visible.get("public_state_available", True) or float(visible.get("state_target_confidence", 0.0)) < 0.55:
                return output(None, source="state_unavailable")
            return output({"action_type": "apply_local_damping", "region_id": int(visible["state_target_hint"])}, source="state")
        if name == "mask_only":
            mask = model_input.get("actionability_mask") or {}
            for info in mask.values():
                if info.get("indirect_target_region") is not None:
                    return output({"action_type": "indirect_stabilize", "region_id": int(info["indirect_target_region"])}, source="mask")
            return output(None, source="mask_hidden")
        if name == "trace_only":
            region = trace.get("region")
            return output(None if region is None else {"action_type": "apply_local_damping", "region_id": int(region)}, source="trace")
        if name == "random":
            region = int(visible.get("state_target_hint", 0))
            return output(None if int(model_input["episode_id"]) % 4 == 0 else {"action_type": "apply_local_damping", "region_id": region}, source="random")
        if name == "always_abstain":
            return output(None, source="always_abstain")
        if name == "conservative_uncertainty":
            if float(trace.get("confidence", 0.0)) < 0.80 or not visible.get("public_state_available", True):
                return output(None, True, source="conservative_uncertainty")
            return output({"action_type": "apply_local_damping", "region_id": int(trace["region"])}, source="conservative_trace")
        if name == "oracle":
            expected = episode["oracle_baseline_view"]["expected_action"]
            out = output(expected, bool(episode["oracle_baseline_view"]["expected_inspect"]), source="oracle")
            if expected and expected.get("action_type") == "indirect_stabilize":
                out["delayed_credit"] = {"credit_assigned": True, "buffer_used": True}
                out["hidden_indirect"] = {"success": True, "candidate_search_used": False, "discovery_source": "oracle"}
            return out
        raise KeyError(name)

    return run


def policy_for(name: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    if name == "b64_transfer_policy":
        return b64_transfer_policy
    return baseline_policy(name)


def audit_policy_integrity(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source = inspect.getsource(b64_transfer_policy) + inspect.getsource(assign_delayed_credit) + inspect.getsource(discover_indirect_from_history)
    forbidden = ["evaluator_ground_truth", "oracle_baseline_view", "expected_decision", "metadata"]
    forbidden_count = sum(1 for token in forbidden if token in source)
    original = b64_transfer_policy(episode, config)
    poisoned = {
        **episode,
        "evaluator_ground_truth": {**episode["evaluator_ground_truth"], "expected_action": None, "condition": "poisoned"},
        "metadata": {**episode["metadata"], "condition": "poisoned"},
    }
    poisoned_output = b64_transfer_policy(poisoned, config)
    return {
        "forbidden_reference_count": forbidden_count,
        "poisoned_evaluator_invariance_pass": comparable_output(original) == comparable_output(poisoned_output),
        "policy_uses_model_input_only": forbidden_count == 0,
    }


def comparable_output(output: dict[str, Any]) -> tuple[Any, ...]:
    action = output.get("action") or {}
    return action.get("action_type"), action.get("region_id"), bool(output.get("inspect")), output.get("transfer_source")

