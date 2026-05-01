from __future__ import annotations

import copy
from typing import Any


GOAL_CODES = {
    "recurrent_goal": [1.0, 0.0, 0.0],
    "field_goal": [0.0, 1.0, 0.0],
    "schema_goal": [0.0, 0.0, 1.0],
    "recurrent": [1.0, 0.0, 0.0],
    "field": [0.0, 1.0, 0.0],
    "schema": [0.0, 0.0, 1.0],
}

GOAL_TO_FIELD = {
    "recurrent_goal": "recurrent_inspect_region",
    "field_goal": "field_inspect_region",
    "schema_goal": "schema_inspect_region",
}


def attach_goal_code(episode: dict[str, Any], goal_family: str, config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(episode)
    canonical = canonical_goal_family(goal_family)
    code = goal_code_for_family(canonical, config)
    expected = expected_region_for_goal(updated, canonical)
    updated["goal_code"] = list(code)
    updated["ground_truth"]["goal_family"] = canonical
    updated["ground_truth"]["oracle_best_inspect_region"] = int(expected)
    updated["ground_truth"]["true_trace_region"] = int(expected)
    updated["ground_truth"]["combined_inspect_region"] = int(expected)
    return updated


def goal_family_from_code(goal_code: list[float]) -> str:
    if not goal_code:
        return "recurrent_goal"
    idx = max(range(len(goal_code)), key=lambda i: float(goal_code[i]))
    return ["recurrent_goal", "field_goal", "schema_goal"][min(idx, 2)]


def expected_region_for_goal(episode: dict[str, Any], goal_family: str) -> int:
    canonical = canonical_goal_family(goal_family)
    field = GOAL_TO_FIELD[canonical]
    return int(episode["ground_truth"][field])


def goal_code_for_family(goal_family: str, config: dict[str, Any] | None = None) -> list[float]:
    canonical = canonical_goal_family(goal_family)
    configured = (config or {}).get("b32", {}).get("goal_codes", {})
    if canonical == "recurrent_goal":
        return list(configured.get("recurrent_goal", GOAL_CODES[canonical]))
    if canonical == "field_goal":
        return list(configured.get("field_goal", GOAL_CODES[canonical]))
    return list(configured.get("schema_goal", GOAL_CODES[canonical]))


def canonical_goal_family(goal_family: str) -> str:
    value = str(goal_family)
    if value in {"recurrent", "recurrent_memory", "recurrent_goal"}:
        return "recurrent_goal"
    if value in {"field", "field_trace", "field_goal"}:
        return "field_goal"
    if value in {"schema", "schema_memory", "schema_goal"}:
        return "schema_goal"
    return "recurrent_goal"
