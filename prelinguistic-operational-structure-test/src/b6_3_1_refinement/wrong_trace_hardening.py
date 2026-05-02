from __future__ import annotations

from typing import Any


WRONG_TRACE_CONDITIONS = [
    "wrong_trace_no_public_state",
    "wrong_trace_history_conflict",
    "wrong_trace_feedback_repair_required",
    "wrong_trace_inspection_required",
]


def apply_wrong_trace_condition(episode: dict[str, Any], condition: str) -> None:
    model_input = episode["model_input"]
    visible = model_input["visible_state"]
    evaluator = episode["evaluator_ground_truth"]
    target = int(evaluator["target_region"])
    wrong = int(evaluator["wrong_target_region"])
    state_decoy = int(evaluator["state_decoy_region"])
    feedback_decoy = int(evaluator["feedback_decoy_region"])

    model_input["previous_trace_state"] = {
        "region": wrong,
        "confidence": 0.92,
        "candidate_regions": [wrong, target],
        "trace_mode": "wrong",
    }
    visible["public_state_available"] = True
    visible["state_target_hint"] = target
    visible["state_target_confidence"] = 0.82
    visible["history_supported_region"] = None
    visible["history_confidence"] = 0.0
    visible["feedback_history_region"] = None
    visible["feedback_confidence"] = 0.0
    visible["inspection_observation_region"] = None
    visible["inspection_confidence"] = 0.0
    evaluator["expected_action"] = {"action_type": "apply_local_damping", "region_id": target}
    evaluator["expected_inspect"] = False
    evaluator["required_repair_source"] = "state"

    if condition == "wrong_trace_no_public_state":
        visible["public_state_available"] = False
        visible["state_target_hint"] = state_decoy
        visible["state_target_confidence"] = 0.0
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.74
        visible["feedback_history_region"] = target
        visible["feedback_confidence"] = 0.70
        evaluator["required_repair_source"] = "history_or_feedback"
    elif condition == "wrong_trace_history_conflict":
        visible["state_target_hint"] = state_decoy
        visible["state_target_confidence"] = 0.62
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.88
        visible["feedback_history_region"] = feedback_decoy
        visible["feedback_confidence"] = 0.54
        evaluator["required_repair_source"] = "history"
    elif condition == "wrong_trace_feedback_repair_required":
        visible["public_state_available"] = False
        visible["state_target_hint"] = state_decoy
        visible["state_target_confidence"] = 0.0
        visible["history_supported_region"] = None
        visible["history_confidence"] = 0.0
        visible["feedback_history_region"] = target
        visible["feedback_confidence"] = 0.90
        evaluator["required_repair_source"] = "feedback"
    elif condition == "wrong_trace_inspection_required":
        visible["state_target_hint"] = state_decoy
        visible["state_target_confidence"] = 0.45
        visible["history_supported_region"] = None
        visible["history_confidence"] = 0.0
        visible["feedback_history_region"] = None
        visible["feedback_confidence"] = 0.0
        visible["inspection_observation_region"] = target
        visible["inspection_confidence"] = 0.92
        evaluator["expected_inspect"] = True
        evaluator["required_repair_source"] = "inspect"


def trace_conflict_resolution_source(model_input: dict[str, Any], disabled: set[str] | None = None) -> dict[str, Any]:
    disabled = disabled or set()
    visible = model_input.get("visible_state", {})
    trace = model_input.get("previous_trace_state", {})
    trace_region = trace.get("region")
    trace_confidence = float(trace.get("confidence", 0.0))
    evidence: list[tuple[str, int, float]] = []

    if "feedback" not in disabled and visible.get("feedback_history_region") is not None:
        evidence.append(("feedback", int(visible["feedback_history_region"]), float(visible.get("feedback_confidence", 0.0))))
    if "history" not in disabled and visible.get("history_supported_region") is not None:
        evidence.append(("history", int(visible["history_supported_region"]), float(visible.get("history_confidence", 0.0))))
    if "inspection" not in disabled and visible.get("inspection_observation_region") is not None:
        evidence.append(("inspect", int(visible["inspection_observation_region"]), float(visible.get("inspection_confidence", 0.0))))
    if "public_state" not in disabled and visible.get("public_state_available", True):
        evidence.append(("state", int(visible.get("state_target_hint", 0)), float(visible.get("state_target_confidence", 0.0))))

    best = max(evidence, key=lambda row: row[2], default=("none", trace_region, 0.0))
    best_source, best_region, best_confidence = best
    conflict = trace_region is None or (best_region is not None and trace_region != best_region and best_confidence >= 0.55)
    confidence_after = min(trace_confidence, 0.45) if conflict else trace_confidence
    repair_region = best_region if conflict and best_confidence >= 0.65 else trace_region
    repair_source = best_source if repair_region == best_region and best_source != "none" else "trace"
    return {
        "trace_conflict_detected": bool(conflict),
        "trace_confidence_before": trace_confidence,
        "trace_confidence_after": confidence_after,
        "trace_supported_region": trace_region,
        "repair_candidate_region": repair_region,
        "repair_source": repair_source,
        "repair_confidence": best_confidence if repair_source != "trace" else trace_confidence,
        "evidence_count": len(evidence),
    }

