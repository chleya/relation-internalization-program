from __future__ import annotations

import inspect
from typing import Any, Callable


POLICY_NAMES = [
    "b64_1_policy_reference",
    "b64_2_combined_policy",
    "state_only",
    "mask_only",
    "trace_only",
    "random",
    "always_abstain",
    "conservative_uncertainty",
    "oracle",
]


def b642_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_input = episode["model_input"]
    visible = model_input.get("visible_state", {})
    action = None
    source = "abstain_uncertain"
    delayed = assign_delayed_credit(model_input)
    hidden = discover_hidden_indirect(model_input)
    if delayed is not None:
        action = delayed
        source = "delayed_credit"
    elif hidden is not None:
        action = hidden
        source = "candidate_search_history"
    else:
        for key, conf_key, label in [
            ("inspection_observation_region", "inspection_confidence", "inspection_recovery"),
            ("fallback_risk_region", "fallback_risk_confidence", "fallback_risk"),
            ("trace_repair_region", "trace_repair_confidence", "trace_repair"),
            ("transition_consistency_region", "transition_confidence", "transition_history"),
            ("feedback_history_region", "feedback_confidence", "feedback_history"),
            ("history_supported_region", "history_confidence", "history"),
        ]:
            if visible.get(key) is not None and float(visible.get(conf_key, 0.0)) >= 0.66:
                action = {"action_type": "apply_local_damping", "region_id": int(visible[key])}
                source = label
                break
    return {
        "policy_name": "b64_2_combined_policy",
        "inspect": source == "inspection_recovery",
        "action": action,
        "transfer_source": source,
        "delayed_credit": {"credit_assigned": delayed is not None, "buffer_used": delayed is not None},
        "hidden_indirect": {"success": hidden is not None, "candidate_search_used": hidden is not None},
        "provenance": {"policy_uses_model_input_only": True, "oracle_value_used": False},
    }


def assign_delayed_credit(model_input: dict[str, Any]) -> dict[str, Any] | None:
    visible = model_input.get("visible_state", {})
    for pending in visible.get("pending_indirect_actions", []):
        for outcome in visible.get("outcome_history", []):
            if outcome.get("source_action_id") == pending.get("action_id") and outcome.get("signature") == pending.get("expected_signature"):
                return {"action_type": "indirect_stabilize", "region_id": int(pending["region_id"])}
    return None


def discover_hidden_indirect(model_input: dict[str, Any]) -> dict[str, Any] | None:
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
            "hidden_indirect": {"success": False, "candidate_search_used": False},
            "provenance": {"policy_uses_model_input_only": True},
        }

    def run(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        model_input = episode["model_input"]
        visible = model_input.get("visible_state", {})
        trace = model_input.get("previous_trace_state", {})
        if name == "b64_1_policy_reference":
            if visible.get("public_state_available", False) and float(visible.get("state_target_confidence", 0.0)) >= 0.55:
                return output({"action_type": "apply_local_damping", "region_id": int(visible["state_target_hint"])}, "b64_1_state")
            if trace.get("region") is not None and float(trace.get("confidence", 0.0)) >= 0.55:
                return output({"action_type": "apply_local_damping", "region_id": int(trace["region"])}, "b64_1_trace")
            return output(None, "b64_1_abstain")
        if name == "state_only":
            if not visible.get("public_state_available", False) or float(visible.get("state_target_confidence", 0.0)) < 0.55:
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
            if float(trace.get("confidence", 0.0)) < 0.80 or not visible.get("public_state_available", False):
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
    if name == "b64_2_combined_policy":
        return b642_policy
    return baseline_policy(name)


def audit_policy_integrity(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source = inspect.getsource(b642_policy) + inspect.getsource(assign_delayed_credit) + inspect.getsource(discover_hidden_indirect)
    forbidden = ["evaluator_ground_truth", "oracle_baseline_view", "expected_decision", "metadata", "failure_source"]
    forbidden_count = sum(1 for token in forbidden if token in source)
    original = b642_policy(episode, config)
    poisoned = {
        **episode,
        "evaluator_ground_truth": {**episode["evaluator_ground_truth"], "expected_action": None, "condition": "poisoned", "failure_source": "poisoned"},
        "metadata": {**episode["metadata"], "condition": "poisoned"},
    }
    changed = b642_policy(poisoned, config)
    return {
        "forbidden_reference_count": forbidden_count,
        "poisoned_evaluator_invariance_pass": comparable(original) == comparable(changed),
        "policy_uses_model_input_only": forbidden_count == 0,
    }


def comparable(output: dict[str, Any]) -> tuple[Any, ...]:
    action = output.get("action") or {}
    return action.get("action_type"), action.get("region_id"), output.get("transfer_source")
