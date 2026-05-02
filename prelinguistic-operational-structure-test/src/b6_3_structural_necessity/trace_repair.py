from __future__ import annotations

from typing import Any


def detect_trace_conflict(model_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    visible = model_input.get("visible_state", {})
    trace = model_input.get("previous_trace_state", {})
    trace_region = trace.get("region")
    trace_confidence = float(trace.get("confidence", 0.0))

    evidence: dict[int, float] = {}
    state_region = visible.get("state_target_hint")
    if state_region is not None:
        evidence[int(state_region)] = evidence.get(int(state_region), 0.0) + float(visible.get("state_target_confidence", 0.0))
    history_region = visible.get("history_supported_region")
    if history_region is not None:
        evidence[int(history_region)] = evidence.get(int(history_region), 0.0) + float(visible.get("history_confidence", 0.0))
    feedback_region = visible.get("feedback_history_region")
    if feedback_region is not None:
        evidence[int(feedback_region)] = evidence.get(int(feedback_region), 0.0) + float(visible.get("feedback_confidence", 0.0))
    for candidate in visible.get("state_target_candidates", []):
        region = candidate.get("region_id")
        if region is not None:
            evidence[int(region)] = evidence.get(int(region), 0.0) + float(candidate.get("support", 0.0))

    supported_region = max(evidence, key=evidence.get) if evidence else None
    supported_value = evidence.get(supported_region, 0.0) if supported_region is not None else 0.0
    mismatch = trace_region is not None and supported_region is not None and int(trace_region) != int(supported_region) and supported_value >= 0.55
    low_confidence = trace_region is None or trace_confidence < 0.55
    transition_conflict = bool(float(visible.get("transition_consistency_score", 0.0)) >= 0.65 and mismatch)
    conflict = bool(low_confidence or mismatch or transition_conflict)

    repair_source = "none"
    repair_candidate = None
    repair_confidence = 0.0
    if supported_region is not None and supported_value >= 0.72:
        repair_candidate = int(supported_region)
        repair_confidence = min(1.0, supported_value)
        if feedback_region == supported_region:
            repair_source = "feedback"
        elif history_region == supported_region:
            repair_source = "history"
        else:
            repair_source = "state"

    return {
        "trace_conflict_detected": conflict,
        "trace_confidence_before": trace_confidence,
        "trace_confidence_after": min(trace_confidence, 0.35) if conflict else trace_confidence,
        "trace_supported_region": None if trace_region is None else int(trace_region),
        "state_supported_region": None if state_region is None else int(state_region),
        "history_supported_region": None if history_region is None else int(history_region),
        "feedback_supported_region": None if feedback_region is None else int(feedback_region),
        "repair_candidate_region": repair_candidate,
        "repair_source": repair_source,
        "repair_confidence": repair_confidence,
        "conflict_evidence": {
            "supported_value": supported_value,
            "mismatch": mismatch,
            "low_confidence": low_confidence,
            "transition_conflict": transition_conflict,
            "evidence_regions": sorted(evidence),
        },
    }


def repair_trace_or_request_inspection(model_input: dict[str, Any], conflict: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trace = dict(model_input.get("previous_trace_state", {}))
    flags = model_input.get("ablation_flags", {})
    risk = float(model_input.get("visible_state", {}).get("risk_history_score", 0.0))
    repaired = dict(trace)
    action = "no_change"
    reason = "trace_consistent"
    inspection_requested = False
    abstain_recommended = False

    if conflict["trace_conflict_detected"]:
        repaired["confidence"] = conflict["trace_confidence_after"]
        repaired["conflict_detected"] = True
        action = "downgrade"
        reason = "trace_conflict_or_low_confidence"
        if not flags.get("inspection_recovery_disabled", False):
            inspection_requested = True
            action = "inspect"
        candidate = conflict.get("repair_candidate_region")
        if candidate is not None and conflict.get("repair_confidence", 0.0) >= 0.72:
            repaired["region"] = int(candidate)
            repaired["confidence"] = max(float(repaired.get("confidence", 0.0)), conflict["repair_confidence"])
            repaired["repair_source"] = conflict["repair_source"]
            action = "repair"
            reason = f"repair_from_{conflict['repair_source']}"
        elif risk >= 0.60:
            abstain_recommended = True
            action = "abstain"
            reason = "uncertain_trace_with_risk"
    else:
        repaired["conflict_detected"] = False
        repaired["repair_source"] = "none"

    return {
        "repaired_trace_state": repaired,
        "inspection_requested": inspection_requested,
        "abstain_recommended": abstain_recommended,
        "repair_action": action,
        "repair_reason": reason,
    }
