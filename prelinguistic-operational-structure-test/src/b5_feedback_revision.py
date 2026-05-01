from __future__ import annotations

import copy
from typing import Any

from .b5_epistemic_pragmatic_values import compute_pragmatic_value


def observe_consequence(episode: dict[str, Any], intervention_action: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    value = compute_pragmatic_value(episode, intervention_action, config)
    return {
        "consequence_value": float(value),
        "observed_trace_region": int(episode["ground_truth"]["true_trace_region"] if value > 0.0 else episode["ground_truth"]["wrong_inspect_region"]),
        "action_success": bool(value > 0.0),
    }


def revise_trace_after_feedback(
    trace_state: dict[str, Any],
    consequence: dict[str, Any],
    episode: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    revised = copy.deepcopy(trace_state)
    revised["region"] = int(consequence["observed_trace_region"])
    revised["uncertainty"] = 0.05 if consequence.get("action_success", False) else 0.35
    revised["confidence"] = 1.0 - float(revised["uncertainty"])
    revised["source"] = "feedback_revised_trace"
    return revised


def feedback_revision_accuracy(revised_trace: dict[str, Any], episode: dict[str, Any]) -> float:
    return 1.0 if int(revised_trace.get("region", -1)) == int(episode["ground_truth"]["true_trace_region"]) else 0.0
