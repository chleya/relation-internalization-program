from __future__ import annotations

from typing import Any


ACTION_TYPES = [
    "do_nothing",
    "inspect_only",
    "apply_local_damping",
    "apply_local_push",
    "block_force_region",
    "stabilize_trace_region",
]


def enumerate_candidate_actions(config: dict[str, Any]) -> list[dict[str, Any]]:
    env = config.get("env", {})
    b4 = config.get("b4", {})
    grid_size = int(b4.get("grid_size", env.get("grid_size", 8)))
    strength = float(b4.get("action_strength", 1.0))
    return [
        {"action_type": action_type, "region_id": int(region), "strength": strength}
        for action_type in action_types(config)
        for region in range(grid_size * grid_size)
    ]


def is_valid_action(action: dict[str, Any], config: dict[str, Any]) -> bool:
    env = config.get("env", {})
    b4 = config.get("b4", {})
    grid_size = int(b4.get("grid_size", env.get("grid_size", 8)))
    try:
        region = int(action.get("region_id", -1))
    except (TypeError, ValueError):
        return False
    return str(action.get("action_type", "")) in action_types(config) and 0 <= region < grid_size * grid_size


def action_cost(action: dict[str, Any], config: dict[str, Any]) -> float:
    return 0.0 if str(action.get("action_type", "")) == "do_nothing" else 1.0


def action_family(action: dict[str, Any]) -> str:
    action_type = str(action.get("action_type", ""))
    if action_type in {"apply_local_damping", "stabilize_trace_region"}:
        return "recurrent"
    if action_type == "block_force_region":
        return "field"
    if action_type == "apply_local_push":
        return "schema"
    if action_type == "inspect_only":
        return "inspect"
    return "none"


def action_types(config: dict[str, Any]) -> list[str]:
    configured = config.get("b4", {}).get("action_space")
    if configured:
        return [str(item) for item in configured]
    return list(ACTION_TYPES)

