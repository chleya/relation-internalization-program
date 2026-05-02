from __future__ import annotations

from typing import Any

from .delayed_credit_assignment import delayed_indirect_policy, select_delayed_indirect_candidate
from .trace_repair import detect_trace_state_conflict, repair_or_downgrade_trace


def b62_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_input = episode["model_input"]
    visible = model_input["visible_state"]
    trace = model_input.get("previous_trace_state", {})
    conflict = detect_trace_state_conflict(model_input, config)
    repair = repair_or_downgrade_trace(model_input, conflict, config)
    repaired_trace = repair["repaired_trace_state"]
    repair_meta = repair["repair_metadata"]
    trace_region = repaired_trace.get("region")
    trace_confidence = float(repaired_trace.get("confidence", 0.0))
    state_target = int(visible["state_target_hint"])
    if trace_region is None or trace_confidence < 0.55:
        target = int(conflict.get("repair_candidate_region") or state_target)
        target_source = "state_fallback"
        inspect = True
    else:
        target = int(trace_region)
        target_source = "trace_repair" if repair_meta["trace_repaired"] else "trace"
        inspect = bool(repair_meta["should_inspect"] or (visible.get("inspect_cost", 0.0) <= 0.05 and trace_confidence < 0.75))

    risk = estimate_risk(model_input, target)
    irreversible = estimate_irreversible(model_input, target)
    safe_alternative = choose_safe_alternative(model_input)
    failure_reason = ""
    if irreversible:
        action = None
        failure_reason = "irreversible_risk"
    elif risk >= 0.72:
        if visible.get("reward_conflict_mode") and safe_alternative is not None:
            action = safe_alternative
            failure_reason = ""
        else:
            delayed_choice = delayed_indirect_policy(model_input, target, config)
            action = delayed_choice["action"]
            delayed_credit = delayed_choice["credit"]
            if action is None:
                failure_reason = "no_indirect_candidate"
            elif delayed_credit.get("backfire") and float(visible.get("backfire_estimate", 0.0)) > 0.45:
                action = None
                failure_reason = "delayed_backfire_risk"
            else:
                failure_reason = ""
    else:
        action = {"action_type": "apply_local_damping", "region_id": target}
        delayed_credit = {"credit_assigned": False, "no_effect": False, "backfire": False, "outcome_type": "not_indirect"}

    if action and action["action_type"] == "indirect_stabilize" and visible.get("backfire_estimate", 0.0) > 0.45:
        action = None
        failure_reason = "delayed_backfire_risk"
    if action is None and risk < 0.72 and repair_meta["should_abstain_if_risky"]:
        failure_reason = failure_reason or "trace_uncertainty"
    return {
        "policy_name": "b62_policy",
        "inspect": bool(inspect),
        "action": action,
        "abstained": action is None,
        "target_source": target_source,
        "trace_confidence": trace_confidence,
        "risk_estimate": risk,
        "irreversible_estimate": irreversible,
        "trace_repair": repair_meta,
        "trace_conflict": conflict,
        "delayed_credit": delayed_credit if "delayed_credit" in locals() else {"credit_assigned": False, "no_effect": False, "backfire": False},
        "failure_reason": failure_reason,
        "provenance": {"policy_uses_model_input_only": True, "oracle_value_used": False},
    }


def estimate_risk(model_input: dict[str, Any], target: int) -> float:
    visible = model_input["visible_state"]
    risk = float(visible.get("risk_history_score", 0.0))
    mask = model_input.get("actionability_mask")
    if mask and target in mask:
        info = mask[target]
        if info.get("unsafe"):
            risk = max(risk, 0.95)
        risk = max(risk, float(info.get("risk_cost", 0.0)))
    if visible.get("latent_risk_marker") == "unsafe":
        risk = max(risk, 0.80)
    return risk


def estimate_irreversible(model_input: dict[str, Any], target: int) -> bool:
    visible = model_input["visible_state"]
    if visible.get("latent_risk_marker") == "irreversible":
        return True
    mask = model_input.get("actionability_mask")
    if mask and target in mask and mask[target].get("irreversible"):
        return True
    return False


def choose_indirect(model_input: dict[str, Any], target: int, config: dict[str, Any]) -> dict[str, Any] | None:
    return select_delayed_indirect_candidate(model_input, target, config)


def choose_safe_alternative(model_input: dict[str, Any]) -> dict[str, Any] | None:
    alternatives = list(model_input.get("visible_state", {}).get("safe_alternative_candidates", []))
    viable = [
        row
        for row in alternatives
        if float(row.get("risk_estimate", 1.0)) <= 0.25 and float(row.get("reward", 0.0)) >= 0.45
    ]
    if not viable:
        return None
    best = max(viable, key=lambda row: float(row.get("reward", 0.0)) + float(row.get("support", 0.0)))
    return {"action_type": "apply_local_damping", "region_id": int(best["region_id"])}
