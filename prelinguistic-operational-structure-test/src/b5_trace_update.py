from __future__ import annotations

import copy
from typing import Any


def export_trace_state(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(episode.get("trace_state", {}))
    state.setdefault("region", int(episode["ground_truth"].get("initial_trace_region", 0)))
    state.setdefault("uncertainty", 0.5)
    state.setdefault("confidence", 1.0 - float(state["uncertainty"]))
    return state


def update_trace_after_inspection(
    trace_state: dict[str, Any],
    inspection_result: dict[str, Any],
    episode: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    updated = copy.deepcopy(trace_state)
    if bool(inspection_result.get("reveals_trace", False)):
        updated["region"] = int(inspection_result["observed_trace_region"])
        updated["uncertainty"] = 0.05
        updated["confidence"] = 0.95
        updated["source"] = "inspection_updated_trace"
    else:
        updated["uncertainty"] = min(1.0, float(updated.get("uncertainty", 0.5)) + 0.10)
        updated["confidence"] = max(0.0, 1.0 - float(updated["uncertainty"]))
        updated["source"] = "unchanged_after_uninformative_inspection"
    return updated


def trace_update_accuracy(updated_trace: dict[str, Any], episode: dict[str, Any]) -> float:
    return 1.0 if int(updated_trace.get("region", -1)) == int(episode["ground_truth"]["trace_after_inspection_region"]) else 0.0


def trace_uncertainty_reduction(before_trace: dict[str, Any], after_trace: dict[str, Any]) -> float:
    return max(0.0, float(before_trace.get("uncertainty", 0.0)) - float(after_trace.get("uncertainty", 0.0)))
