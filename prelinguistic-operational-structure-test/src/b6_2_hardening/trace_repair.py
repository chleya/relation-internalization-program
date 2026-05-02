from __future__ import annotations

from typing import Any


def detect_trace_state_conflict(model_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Compare the public previous trace with model-visible state/history cues.

    This function intentionally reads only model_input. It does not inspect
    evaluator_ground_truth, metadata condition labels, oracle views, or answer keys.
    """

    visible = model_input.get("visible_state", {})
    trace = model_input.get("previous_trace_state", {})
    trace_region = trace.get("region")
    trace_confidence = float(trace.get("confidence", 0.0))
    state_region = visible.get("state_target_hint")
    state_confidence = float(visible.get("state_target_confidence", 0.0))
    feedback_region = visible.get("feedback_history_region")
    feedback_confidence = float(visible.get("feedback_confidence", 0.0))
    transition_consistency = float(visible.get("transition_consistency_score", 0.0))

    evidence: dict[int, float] = {}
    if state_region is not None:
        evidence[int(state_region)] = evidence.get(int(state_region), 0.0) + state_confidence
    if feedback_region is not None:
        evidence[int(feedback_region)] = evidence.get(int(feedback_region), 0.0) + feedback_confidence
    for candidate in visible.get("state_target_candidates", []):
        region = candidate.get("region_id")
        if region is not None:
            evidence[int(region)] = evidence.get(int(region), 0.0) + float(candidate.get("support", 0.0))

    if evidence:
        state_supported_region = max(evidence, key=evidence.get)
        state_support = evidence[state_supported_region]
    else:
        state_supported_region = None
        state_support = 0.0

    trace_supported_region = None if trace_region is None else int(trace_region)
    region_mismatch = (
        state_supported_region is not None
        and trace_supported_region is not None
        and state_supported_region != trace_supported_region
        and state_support >= 0.55
    )
    low_confidence = trace_supported_region is None or trace_confidence < 0.55
    inconsistent_transition = transition_consistency >= 0.65 and region_mismatch
    conflict = bool(low_confidence or region_mismatch or inconsistent_transition)
    repair_candidate = state_supported_region if state_support >= 0.70 else None

    return {
        "trace_conflict_detected": conflict,
        "trace_confidence": trace_confidence,
        "state_supported_region": state_supported_region,
        "trace_supported_region": trace_supported_region,
        "repair_candidate_region": repair_candidate,
        "conflict_evidence": {
            "state_support": state_support,
            "feedback_confidence": feedback_confidence,
            "transition_consistency_score": transition_consistency,
            "low_confidence": low_confidence,
            "region_mismatch": region_mismatch,
            "inconsistent_transition": inconsistent_transition,
        },
    }


def repair_or_downgrade_trace(model_input: dict[str, Any], conflict: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Lower confidence when trace and public evidence disagree.

    Strong public evidence can propose a repair candidate, but this is recorded
    explicitly so B6.2 cannot claim private-trace-only correction.
    """

    trace = dict(model_input.get("previous_trace_state", {}))
    original_region = trace.get("region")
    original_confidence = float(trace.get("confidence", 0.0))
    repaired = dict(trace)
    repaired_region = original_region
    repaired_by_public_state = False
    should_inspect = False
    should_abstain_if_risky = False

    if conflict.get("trace_conflict_detected"):
        repaired["confidence"] = min(original_confidence, 0.35)
        repaired["conflict_detected"] = True
        should_inspect = True
        candidate = conflict.get("repair_candidate_region")
        if candidate is not None:
            repaired_region = int(candidate)
            repaired["region"] = repaired_region
            repaired["confidence"] = max(float(repaired["confidence"]), 0.62)
            repaired["repair_source"] = "public_state_feedback"
            repaired_by_public_state = True
        else:
            should_abstain_if_risky = True
            repaired["repair_source"] = "downgrade_only"
    else:
        repaired["conflict_detected"] = False
        repaired["repair_source"] = "none"

    return {
        "repaired_trace_state": repaired,
        "repair_metadata": {
            "original_region": original_region,
            "repaired_region": repaired_region,
            "trace_confidence_downgraded": float(repaired.get("confidence", 0.0)) < original_confidence,
            "trace_repaired": repaired_region != original_region,
            "repair_used_public_state": repaired_by_public_state,
            "should_inspect": should_inspect,
            "should_abstain_if_risky": should_abstain_if_risky,
        },
    }
