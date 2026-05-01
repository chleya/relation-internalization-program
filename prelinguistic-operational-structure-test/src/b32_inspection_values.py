from __future__ import annotations

from typing import Any

import numpy as np

from .b32_goal_conditioning import goal_family_from_code


VALUE_FIELDS = {
    "recurrent_goal": ("recurrent_value", "recurrent_inspect_region"),
    "field_goal": ("field_value", "field_inspect_region"),
    "schema_goal": ("schema_value", "schema_inspect_region"),
}


def compute_recurrent_inspection_value(episode: dict[str, Any], region_id: int, config: dict[str, Any]) -> float:
    return family_value(episode, "recurrent_goal", region_id)


def compute_field_inspection_value(episode: dict[str, Any], region_id: int, config: dict[str, Any]) -> float:
    return family_value(episode, "field_goal", region_id)


def compute_schema_inspection_value(episode: dict[str, Any], region_id: int, config: dict[str, Any]) -> float:
    return family_value(episode, "schema_goal", region_id)


def compute_combined_inspection_value(episode: dict[str, Any], region_id: int, goal_code: list[float], config: dict[str, Any]) -> float:
    weights = np.asarray(goal_code or [1.0, 0.0, 0.0], dtype=np.float32)
    if float(weights.sum()) <= 0.0:
        weights = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    weights = weights / float(weights.sum())
    return float(
        weights[0] * compute_recurrent_inspection_value(episode, region_id, config)
        + weights[1] * compute_field_inspection_value(episode, region_id, config)
        + weights[2] * compute_schema_inspection_value(episode, region_id, config)
    )


def build_family_value_maps(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, dict[int, float]]:
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    regions = list(range(grid_size * grid_size))
    recurrent = {region: target_value(episode, "recurrent_goal", region) for region in regions}
    field = {region: target_value(episode, "field_goal", region) for region in regions}
    schema = {region: target_value(episode, "schema_goal", region) for region in regions}
    goal_code = list(episode.get("goal_code", [1.0, 0.0, 0.0]))
    combined = {region: compute_combined_from_maps(region, goal_code, recurrent, field, schema) for region in regions}
    return {
        "recurrent_value": recurrent,
        "field_value": field,
        "schema_value": schema,
        "combined_value": combined,
    }


def value_decomposition_rows(episode: dict[str, Any], config: dict[str, Any], episode_id: int = 0) -> list[dict[str, Any]]:
    maps = episode["ground_truth"].get("inspection_values", build_family_value_maps(episode, config))
    rows = []
    for region in sorted(maps["recurrent_value"]):
        rows.append(
            {
                "episode_id": episode_id,
                "region_id": int(region),
                "recurrent_value": float(maps["recurrent_value"][region]),
                "field_value": float(maps["field_value"][region]),
                "schema_value": float(maps["schema_value"][region]),
                "combined_value": float(maps["combined_value"].get(region, 0.0)),
                "best_recurrent_region": best_region(maps["recurrent_value"]),
                "best_field_region": best_region(maps["field_value"]),
                "best_schema_region": best_region(maps["schema_value"]),
                "best_combined_region": best_region(maps["combined_value"]),
            }
        )
    return rows


def family_value(episode: dict[str, Any], goal_family: str, region_id: int) -> float:
    maps = episode["ground_truth"].get("inspection_values", {})
    key, _ = VALUE_FIELDS[goal_family]
    if key in maps:
        return float(maps[key].get(int(region_id), 0.0))
    return target_value(episode, goal_family, region_id)


def target_value(episode: dict[str, Any], goal_family: str, region_id: int) -> float:
    _, target_field = VALUE_FIELDS[goal_family]
    target = int(episode["ground_truth"][target_field])
    region = int(region_id)
    if region == target:
        return 1.0
    if region == int(episode["ground_truth"].get("saliency_region", -1)):
        return 0.05
    return 0.0


def compute_combined_from_maps(
    region: int,
    goal_code: list[float],
    recurrent: dict[int, float],
    field: dict[int, float],
    schema: dict[int, float],
) -> float:
    goal = goal_family_from_code(goal_code)
    if goal == "recurrent_goal":
        return float(recurrent.get(region, 0.0))
    if goal == "field_goal":
        return float(field.get(region, 0.0))
    return float(schema.get(region, 0.0))


def best_region(value_map: dict[int, float]) -> int:
    return int(max(value_map.items(), key=lambda item: (float(item[1]), -int(item[0])))[0]) if value_map else -1
