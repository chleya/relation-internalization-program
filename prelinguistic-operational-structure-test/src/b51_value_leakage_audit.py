from __future__ import annotations

from typing import Any

import numpy as np


FORBIDDEN_B5_KEYS = [
    "oracle_closed_loop_plan",
    "oracle_inspect_region",
    "oracle_intervention_action",
    "best_inspect_region",
    "best_intervention_after_inspection",
    "best_intervention_without_inspection",
    "epistemic_values",
    "pragmatic_values_before_inspection",
    "pragmatic_values_after_inspection",
    "oracle_trace_update_target",
    "oracle_feedback_revision_target",
    "closed_loop_value",
    "oracle_closed_loop_score",
    "ground_truth",
]


def audit_b5_value_leakage(
    model_input: dict[str, Any],
    policy_output: dict[str, Any],
    provenance: dict[str, Any],
    forbidden_keys: list[str] = FORBIDDEN_B5_KEYS,
) -> dict[str, Any]:
    paths = []
    for root_name, structure in [("model_input", model_input), ("policy_output", policy_output), ("provenance", provenance)]:
        paths.extend(find_forbidden_paths(structure, root_name, set(forbidden_keys)))
    return {
        "value_leakage_count": len(paths),
        "oracle_plan_usage_rate": 1.0 if bool(provenance.get("oracle_plan_used", provenance.get("oracle_closed_loop_plan_used", False))) else 0.0,
        "oracle_trace_update_usage_rate": 1.0 if bool(provenance.get("oracle_trace_update_used", False)) else 0.0,
        "oracle_feedback_revision_usage_rate": 1.0 if bool(provenance.get("oracle_feedback_revision_used", False)) else 0.0,
        "forbidden_key_paths": ";".join(paths),
    }


def summarize_b5_value_leakage(leakage_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    return {
        "value_leakage_count": float(sum(int(record.get("value_leakage_count", 0)) for record in leakage_records)),
        "oracle_plan_usage_rate": mean_or_zero([float(record.get("oracle_plan_usage_rate", 0.0)) for record in leakage_records]),
        "oracle_trace_update_usage_rate": mean_or_zero([float(record.get("oracle_trace_update_usage_rate", 0.0)) for record in leakage_records]),
        "oracle_feedback_revision_usage_rate": mean_or_zero([float(record.get("oracle_feedback_revision_usage_rate", 0.0)) for record in leakage_records]),
    }


def find_forbidden_paths(structure: Any, path: str, forbidden: set[str]) -> list[str]:
    paths = []
    if isinstance(structure, dict):
        for key, value in structure.items():
            child_path = f"{path}.{key}"
            if str(key) in forbidden:
                paths.append(child_path)
            paths.extend(find_forbidden_paths(value, child_path, forbidden))
    elif isinstance(structure, list):
        for idx, value in enumerate(structure):
            paths.extend(find_forbidden_paths(value, f"{path}[{idx}]", forbidden))
    return paths


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
