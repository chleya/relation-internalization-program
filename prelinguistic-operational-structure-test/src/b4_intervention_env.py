from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .b32_mechanism_inspection_env import b32_runtime_config, make_family_specific_inspection_episode
from .b4_intervention_values import compute_intervention_value


FAMILY_TO_GOAL = {
    "recurrent": "recurrent_goal",
    "field": "field_goal",
    "schema": "schema_goal",
    "recurrent_goal": "recurrent_goal",
    "field_goal": "field_goal",
    "schema_goal": "schema_goal",
}

FAMILY_ACTION_TYPES = {
    "recurrent": "stabilize_trace_region",
    "field": "block_force_region",
    "schema": "apply_local_push",
}


def make_b4_intervention_episode(
    config: dict[str, Any],
    seed: int,
    episode_type: str,
    family: str | None = None,
) -> dict[str, Any]:
    canonical = canonical_family(family or "recurrent")
    base = make_family_specific_inspection_episode(config, seed, FAMILY_TO_GOAL[canonical])
    episode = copy.deepcopy(base)
    gt = episode["ground_truth"]
    family_actions = {
        "recurrent": make_action("stabilize_trace_region", int(gt["recurrent_inspect_region"]), config),
        "field": make_action("block_force_region", int(gt["field_inspect_region"]), config),
        "schema": make_action("apply_local_push", int(gt["schema_inspect_region"]), config),
    }
    gt.update(
        {
            "episode_type": str(episode_type),
            "b4_family": canonical,
            "true_trace_region": int(gt[f"{canonical}_inspect_region"]),
            "short_horizon_region": int(gt.get("short_horizon_region", gt.get("saliency_region", 0))),
            "recurrent_intervention_region": int(gt["recurrent_inspect_region"]),
            "field_intervention_region": int(gt["field_inspect_region"]),
            "schema_intervention_region": int(gt["schema_inspect_region"]),
            "family_best_actions": family_actions,
            "oracle_best_action": dict(family_actions[canonical]),
            "baseline_outcome_error": 1.0,
            "delay": int(gt.get("delay", 4)),
        }
    )
    gt["intervention_values"] = build_intervention_values(gt)
    return episode


def make_b4_datasets(config: dict[str, Any], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    runtime = b4_runtime_config(config)
    b4 = b4_config(config)
    n_test = effective_count(int(b4.get("n_test", 32)), b4)
    n_ood = effective_count(int(b4.get("n_ood", 32)), b4)
    families = ["recurrent", "field", "schema"]
    return {
        "test": [make_b4_intervention_episode(runtime, seed + idx * 17, "b4_trace_guided_intervention", families[idx % 3]) for idx in range(n_test)],
        "ood": [make_b4_intervention_episode(runtime, seed + 20000 + idx * 23, "b4_delay_ood_intervention", families[idx % 3]) for idx in range(n_ood)],
    }


def apply_intervention_action(episode: dict[str, Any], action: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(episode)
    value = compute_intervention_value(episode, action, config)
    baseline_error = float(episode["ground_truth"].get("baseline_outcome_error", 1.0))
    updated["intervention"] = {
        "action": dict(action),
        "intervention_value": float(value),
        "baseline_outcome_error": baseline_error,
        "intervened_outcome_error": max(0.0, baseline_error - float(value)),
        "dynamics_changed": str(action.get("action_type", "")) not in {"do_nothing", "inspect_only"},
    }
    return updated


def evaluate_intervention_outcome(
    original_episode: dict[str, Any],
    intervened_episode: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    baseline = float(original_episode["ground_truth"].get("baseline_outcome_error", 1.0))
    intervened = float(intervened_episode.get("intervention", {}).get("intervened_outcome_error", baseline))
    return {
        "baseline_outcome_error": baseline,
        "intervened_outcome_error": intervened,
        "outcome_improvement": max(0.0, baseline - intervened),
    }


def make_action(action_type: str, region_id: int, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": str(action_type),
        "region_id": int(region_id),
        "strength": float(config.get("b4", {}).get("action_strength", 1.0)),
    }


def build_intervention_values(gt: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    action = gt["oracle_best_action"]
    values[action_key(action)] = 1.0
    region = int(action["region_id"])
    values[action_key({"action_type": "inspect_only", "region_id": region})] = 0.0
    for action_type in FAMILY_ACTION_TYPES.values():
        if action_type != action["action_type"]:
            values[action_key({"action_type": action_type, "region_id": region})] = 0.35
    return values


def action_key(action: dict[str, Any]) -> str:
    return f"{action.get('action_type')}:{int(action.get('region_id', -1))}"


def canonical_family(family: str) -> str:
    value = str(family)
    if value in {"recurrent", "recurrent_goal", "recurrent_flow_checkpoint_model"}:
        return "recurrent"
    if value in {"field", "field_goal", "field_memory_model"}:
        return "field"
    if value in {"schema", "schema_goal", "schema_memory_model"}:
        return "schema"
    return "recurrent"


def b4_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    base = resolve_b4_base_config(config)
    runtime = copy.deepcopy(base) if "base_config" not in config and "env" in base else b32_runtime_config(base)
    b4 = b4_config(config)
    env = dict(runtime.get("env", {}))
    for key in ("frame_size", "grid_size", "past_frames", "future_frames"):
        if key in b4:
            env[key] = int(b4[key])
    runtime["env"] = env
    runtime["b4"] = b4
    return runtime


def resolve_b4_base_config(config: dict[str, Any]) -> dict[str, Any]:
    if "base_config" not in config:
        return config
    path = Path(str(config.get("base_config", "configs/b32_mechanism_inspection.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def b4_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b4", {"n_test": 32, "n_ood": 32, "gates": {}})


def effective_count(value: int, b4: dict[str, Any]) -> int:
    cap = b4.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))
