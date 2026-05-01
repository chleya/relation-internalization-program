from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .b22_disagreement_env import make_trace_disagreement_episode
from .b3_active_inspection_env import b3_runtime_config
from .b32_goal_conditioning import attach_goal_code
from .b32_inspection_values import build_family_value_maps


def make_family_specific_inspection_episode(
    config: dict[str, Any],
    seed: int,
    dominant_family: str | None = None,
) -> dict[str, Any]:
    base_family = {
        "recurrent_goal": "recurrent",
        "field_goal": "field",
        "schema_goal": "schema",
        "recurrent": "recurrent",
        "field": "field",
        "schema": "schema",
    }.get(str(dominant_family), "recurrent")
    episode = copy.deepcopy(make_trace_disagreement_episode(config, seed, base_family))
    gt = episode["ground_truth"]
    recurrent_region = int(gt["temporal_trace_region"])
    field_region = int(gt["field_trace_region"])
    schema_region = int(gt["schema_trace_region"])
    if len({recurrent_region, field_region, schema_region}) < 3:
        return make_family_specific_inspection_episode(config, seed + 997, dominant_family)
    saliency_region = int(gt.get("early_saliency_region", gt.get("saliency_region", 0)))
    short_horizon_region = saliency_region
    gt.update(
        {
            "episode_type": "b32_family_specific_inspection",
            "recurrent_inspect_region": recurrent_region,
            "field_inspect_region": field_region,
            "schema_inspect_region": schema_region,
            "saliency_region": saliency_region,
            "short_horizon_region": short_horizon_region,
            "dominant_family": str(dominant_family or ""),
        }
    )
    goal_family = dominant_family or "recurrent_goal"
    episode = attach_goal_code(episode, goal_family, config)
    episode["ground_truth"]["inspection_values"] = build_family_value_maps(episode, config)
    return episode


def make_mechanism_disagreement_episode(config: dict[str, Any], seed: int) -> dict[str, Any]:
    episode = make_family_specific_inspection_episode(config, seed, None)
    episode["ground_truth"]["episode_type"] = "b32_mechanism_disagreement"
    return episode


def make_b32_datasets(config: dict[str, Any], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    runtime = b32_runtime_config(config)
    b32 = b32_config(config)
    n_test = effective_count(int(b32.get("n_test", 32)), b32)
    n_ood = effective_count(int(b32.get("n_ood", 32)), b32)
    families = ["recurrent_goal", "field_goal", "schema_goal"]
    return {
        "test": [make_family_specific_inspection_episode(runtime, seed + idx * 17, families[idx % 3]) for idx in range(n_test)],
        "disagreement": [make_mechanism_disagreement_episode(runtime, seed + 10000 + idx * 19) for idx in range(n_test)],
        "ood": [make_family_specific_inspection_episode(runtime, seed + 20000 + idx * 23, families[idx % 3]) for idx in range(n_ood)],
    }


def b32_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    base = resolve_b32_base_config(config)
    runtime = copy.deepcopy(base) if "base_config" not in config and "env" in base else b3_runtime_config(base)
    b32 = b32_config(config)
    env = dict(runtime.get("env", {}))
    for key in ("frame_size", "grid_size", "past_frames", "future_frames"):
        if key in b32:
            env[key] = int(b32[key])
    runtime["env"] = env
    runtime["b32"] = b32
    return runtime


def resolve_b32_base_config(config: dict[str, Any]) -> dict[str, Any]:
    if "base_config" not in config:
        return config
    path = Path(str(config.get("base_config", "configs/b31_inspection_audit.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def b32_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b32", {"n_test": 32, "n_ood": 32, "gates": {}})


def effective_count(value: int, b32: dict[str, Any]) -> int:
    cap = b32.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))
