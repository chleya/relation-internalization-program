from __future__ import annotations

from typing import Any

from src.b6_2_hardening.delayed_credit_assignment import delayed_indirect_policy

from .trace_repair import detect_trace_conflict, repair_trace_or_request_inspection


def b63_policy(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_input = episode["model_input"]
    visible = model_input.get("visible_state", {})
    flags = model_input.get("ablation_flags", {})
    conflict = detect_trace_conflict(model_input, config)
    repair = repair_trace_or_request_inspection(model_input, conflict, config)
    trace = repair["repaired_trace_state"]
    trace_region = trace.get("region")
    trace_confidence = float(trace.get("confidence", 0.0))

    if trace_region is None or trace_confidence < 0.55:
        target = int(visible.get("state_target_hint", 0))
        target_source = "state_fallback"
    else:
        target = int(trace_region)
        target_source = "trace_repair" if repair["repair_action"] == "repair" else "trace"

    inspect = bool(repair["inspection_requested"])
    if flags.get("inspection_recovery_disabled"):
        inspect = False

    risk = estimate_risk(model_input, target)
    irreversible = estimate_irreversible(model_input, target)
    action = None
    failure_reason = ""
    delayed_credit = {"credit_assigned": False, "no_effect": False, "backfire": False, "outcome_type": "not_indirect"}

    if repair["abstain_recommended"]:
        failure_reason = repair["repair_reason"]
    elif irreversible:
        failure_reason = "irreversible_risk"
    elif risk >= 0.72:
        safe = choose_safe_alternative(model_input)
        if visible.get("reward_conflict_mode") and safe is not None:
            action = safe
        else:
            choice = delayed_indirect_policy(model_input, target, config)
            action = choice["action"]
            delayed_credit = choice["credit"]
            if action is None:
                failure_reason = "no_indirect_candidate"
            elif delayed_credit.get("backfire"):
                action = None
                failure_reason = "delayed_backfire_risk"
    else:
        action = {"action_type": "apply_local_damping", "region_id": target}

    return {
        "policy_name": "b63_policy",
        "inspect": inspect,
        "action": action,
        "abstained": action is None,
        "target_source": target_source,
        "risk_estimate": risk,
        "trace_conflict": conflict,
        "trace_repair": {
            "trace_repaired": repair["repair_action"] == "repair",
            "trace_confidence_downgraded": conflict["trace_confidence_after"] < conflict["trace_confidence_before"],
            "repair_used_public_state": repair["repair_action"] == "repair" and conflict["repair_source"] == "state",
            "repair_action": repair["repair_action"],
            "repair_reason": repair["repair_reason"],
        },
        "delayed_credit": delayed_credit,
        "failure_reason": failure_reason,
        "provenance": {"policy_uses_model_input_only": True, "oracle_value_used": False},
    }


def estimate_risk(model_input: dict[str, Any], target: int) -> float:
    visible = model_input.get("visible_state", {})
    risk = float(visible.get("risk_history_score", 0.0))
    marker = visible.get("latent_risk_marker")
    if marker == "unsafe":
        risk = max(risk, 0.82)
    if marker == "irreversible":
        risk = max(risk, 0.92)
    mask = model_input.get("actionability_mask")
    if mask and target in mask:
        info = mask[target]
        if info.get("unsafe"):
            risk = max(risk, 0.95)
        risk = max(risk, float(info.get("risk_cost", 0.0)))
    return risk


def estimate_irreversible(model_input: dict[str, Any], target: int) -> bool:
    visible = model_input.get("visible_state", {})
    if visible.get("latent_risk_marker") == "irreversible":
        return True
    mask = model_input.get("actionability_mask")
    return bool(mask and target in mask and mask[target].get("irreversible"))


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
