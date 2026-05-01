from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .b41_intervention_degeneracy_audit import b41_runtime_config
from .b4_intervention_env import canonical_family, make_b4_intervention_episode


ALLOWED_ACTION_TYPES = [
    "apply_local_damping",
    "apply_local_push",
    "block_force_region",
    "stabilize_trace_region",
]

FAMILY_CONFIG_KEY = {
    "recurrent": "recurrent_memory",
    "field": "field_trace",
    "schema": "schema_memory",
}


def make_action_type_specific_episode(
    config: dict[str, Any],
    seed: int,
    family: str,
    required_action_type: str | None = None,
) -> dict[str, Any]:
    canonical = canonical_family(family)
    episode = copy.deepcopy(make_b4_intervention_episode(config, seed, "b42_action_type_specific", canonical))
    required = required_action_type or expected_action_type_for_family(canonical, episode, config, seed)
    return attach_b42_action_type_fields(episode, canonical, required, config)


def make_action_type_ood_episode(
    config: dict[str, Any],
    seed: int,
    family: str,
    ood_action_type: str,
) -> dict[str, Any]:
    canonical = canonical_family(family)
    episode = copy.deepcopy(make_b4_intervention_episode(config, seed, "b42_action_type_ood", canonical))
    return attach_b42_action_type_fields(episode, canonical, ood_action_type, config)


def make_b42_datasets(config: dict[str, Any], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    runtime = b42_runtime_config(config)
    b42 = b42_config(config)
    n_test = effective_count(int(b42.get("n_test", 32)), b42)
    n_ood = effective_count(int(b42.get("n_ood", 32)), b42)
    families = ["recurrent", "field", "schema"]
    return {
        "test": [make_action_type_specific_episode(runtime, seed + idx * 17, families[idx % 3]) for idx in range(n_test)],
        "ood": [
            make_action_type_ood_episode(runtime, seed + 20000 + idx * 23, families[idx % 3], ood_action_for_family(families[idx % 3], runtime))
            for idx in range(n_ood)
        ],
    }


def attach_b42_action_type_fields(
    episode: dict[str, Any],
    family: str,
    required_action_type: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    allowed = allowed_action_types(config)
    if required_action_type not in allowed:
        required_action_type = allowed[0]
    gt = episode["ground_truth"]
    region = int(gt[f"{family}_intervention_region"])
    wrong_actions = [action_type for action_type in allowed if action_type != required_action_type]
    gt.update(
        {
            "episode_type": gt.get("episode_type", "b42_action_type_specific"),
            "family": family,
            "trace_family": family,
            "true_trace_region": region,
            "intervention_region": region,
            "required_action_type": required_action_type,
            "wrong_action_types": wrong_actions,
            "oracle_best_action": {
                "action_type": required_action_type,
                "region_id": region,
                "strength": float(config.get("b42", config.get("b4", {})).get("action_strength", 1.0)),
            },
        }
    )
    gt["action_type_values"] = build_action_type_values(region, required_action_type, config)
    episode["action_type_signal"] = build_action_type_signal(required_action_type, config)
    return episode


def expected_action_type_for_family(family: str, episode: dict[str, Any], config: dict[str, Any], seed: int | None = None) -> str:
    canonical = canonical_family(family)
    key = FAMILY_CONFIG_KEY[canonical]
    preferred = config.get("b42", {}).get("family_action_map_train", {}).get(key, {}).get("preferred_actions")
    actions = [str(action) for action in preferred] if preferred else default_train_actions(canonical)
    seed_marker = int(seed) if seed is not None else int(episode["ground_truth"].get("true_trace_region", 0)) + int(episode["ground_truth"].get("saliency_region", 0))
    return actions[seed_marker % len(actions)]


def ood_action_for_family(family: str, config: dict[str, Any]) -> str:
    canonical = canonical_family(family)
    key = FAMILY_CONFIG_KEY[canonical]
    preferred = config.get("b42", {}).get("family_action_map_ood", {}).get(key, {}).get("preferred_actions")
    actions = [str(action) for action in preferred] if preferred else [default_train_actions(canonical)[-1]]
    return actions[0]


def build_action_type_values(region: int, required_action_type: str, config: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    n_regions = grid_size * grid_size
    for action_type in allowed_action_types(config):
        for region_id in range(n_regions):
            value = 0.0
            if region_id == region and action_type == required_action_type:
                value = 1.0
            elif region_id == region:
                value = 0.10
            elif action_type == required_action_type:
                value = 0.05
            values[action_key(action_type, region_id)] = value
    return values


def build_action_type_signal(required_action_type: str, config: dict[str, Any]) -> dict[str, float]:
    signal = {action_type: 0.05 for action_type in allowed_action_types(config)}
    signal[required_action_type] = 1.0
    return signal


def allowed_action_types(config: dict[str, Any]) -> list[str]:
    actions = config.get("b42", {}).get("allowed_action_types")
    if actions:
        return [str(action) for action in actions]
    return list(ALLOWED_ACTION_TYPES)


def default_train_actions(family: str) -> list[str]:
    canonical = canonical_family(family)
    if canonical == "recurrent":
        return ["apply_local_damping", "stabilize_trace_region"]
    if canonical == "field":
        return ["block_force_region", "apply_local_damping"]
    return ["stabilize_trace_region", "apply_local_push"]


def action_key(action_type: str, region_id: int) -> str:
    return f"{action_type}:{int(region_id)}"


def b42_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    base = resolve_b42_base_config(config)
    runtime = b41_runtime_config(base)
    b42 = b42_config(config)
    env = dict(runtime.get("env", {}))
    for key in ("frame_size", "grid_size", "past_frames", "future_frames"):
        if key in b42:
            env[key] = int(b42[key])
    runtime["env"] = env
    runtime["b42"] = b42
    runtime["b4"] = {**runtime.get("b4", {}), "action_strength": float(b42.get("action_strength", runtime.get("b4", {}).get("action_strength", 1.0)))}
    return runtime


def resolve_b42_base_config(config: dict[str, Any]) -> dict[str, Any]:
    if "base_config" not in config:
        return config
    path = Path(str(config.get("base_config", "configs/b41_intervention_audit.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def b42_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b42", {"n_test": 32, "n_ood": 32, "gates": {}})


def effective_count(value: int, b42: dict[str, Any]) -> int:
    cap = b42.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))
