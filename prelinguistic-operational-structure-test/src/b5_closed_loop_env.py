from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .b4_intervention_env import canonical_family
from .b42_action_type_env import b42_runtime_config, make_action_type_specific_episode


EPISODE_TYPES = [
    "inspect_needed",
    "intervention_clear",
    "no_action",
    "misleading_initial_trace",
]


def make_b5_closed_loop_episode(
    config: dict[str, Any],
    seed: int,
    episode_type: str = "inspect_needed",
    family: str | None = None,
) -> dict[str, Any]:
    canonical = canonical_family(family or ["recurrent", "field", "schema"][seed % 3])
    base = copy.deepcopy(make_action_type_specific_episode(config, seed, canonical))
    gt = base["ground_truth"]
    episode_type = str(episode_type)
    if episode_type not in EPISODE_TYPES:
        episode_type = EPISODE_TYPES[seed % len(EPISODE_TYPES)]

    true_region = int(gt["intervention_region"])
    saliency_region = int(gt.get("saliency_region", (true_region + 1) % 64))
    short_region = int(gt.get("short_horizon_region", saliency_region))
    wrong_region = first_different_region(true_region, saliency_region, short_region, config)
    needs_inspection = episode_type in {"inspect_needed", "misleading_initial_trace"}
    should_do_nothing = episode_type == "no_action"
    should_intervene = not should_do_nothing
    initial_region = wrong_region if needs_inspection else true_region
    uncertainty = 0.85 if needs_inspection else 0.10
    oracle_action = dict(gt["oracle_best_action"])
    if should_do_nothing:
        oracle_action = {"action_type": "do_nothing", "region_id": true_region, "strength": 0.0}

    costs = b5_costs(config)
    gt.update(
        {
            "episode_type": f"b5_{episode_type}",
            "closed_loop_episode_type": episode_type,
            "family": canonical,
            "trace_family": canonical,
            "needs_inspection": needs_inspection,
            "should_intervene_immediately": episode_type == "intervention_clear",
            "should_do_nothing": should_do_nothing,
            "initial_trace_region": initial_region,
            "true_trace_region": true_region,
            "oracle_inspect_region": true_region,
            "wrong_inspect_region": wrong_region,
            "trace_after_inspection_region": true_region,
            "oracle_intervention_action": oracle_action,
            "oracle_closed_loop_plan": {
                "inspect": needs_inspection,
                "inspect_region": true_region if needs_inspection else -1,
                "intervene": should_intervene,
                "intervention_action": oracle_action,
            },
            "epistemic_gain": 1.0 if needs_inspection else 0.0,
            "pragmatic_gain": 1.0 if should_intervene else 0.0,
            "inspection_cost": costs["inspection_cost"],
            "intervention_cost": costs["intervention_cost"],
            "wrong_inspect_penalty": costs["wrong_inspect_penalty"],
            "wrong_intervention_penalty": costs["wrong_intervention_penalty"],
            "expected_closed_loop_value": 1.0,
            "delay": int(gt.get("delay", 4)),
        }
    )
    base["trace_state"] = {
        "family": canonical,
        "region": initial_region,
        "confidence": 1.0 - uncertainty,
        "uncertainty": uncertainty,
        "source": "initial_private_trace",
    }
    base["closed_loop_signal"] = {
        "needs_inspection": needs_inspection,
        "should_intervene": should_intervene,
        "epistemic_scores": {str(true_region): 1.0, str(wrong_region): 0.0},
        "pragmatic_scores": {str(true_region): 1.0, str(wrong_region): 0.0},
        "action_type_signal": dict(base.get("action_type_signal", {})),
    }
    return base


def make_b5_datasets(config: dict[str, Any], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    runtime = b5_runtime_config(config)
    b5 = b5_config(config)
    n_test = effective_count(int(b5.get("n_test", 32)), b5)
    n_ood = effective_count(int(b5.get("n_ood", 32)), b5)
    families = ["recurrent", "field", "schema"]
    return {
        "test": [
            make_b5_closed_loop_episode(runtime, seed + idx * 17, EPISODE_TYPES[idx % len(EPISODE_TYPES)], families[idx % 3])
            for idx in range(n_test)
        ],
        "ood": [
            make_b5_closed_loop_episode(runtime, seed + 20000 + idx * 23, EPISODE_TYPES[(idx + 1) % len(EPISODE_TYPES)], families[idx % 3])
            for idx in range(n_ood)
        ],
    }


def inspect_region(episode: dict[str, Any], region_id: int, config: dict[str, Any]) -> dict[str, Any]:
    gt = episode["ground_truth"]
    correct = int(region_id) == int(gt["oracle_inspect_region"])
    return {
        "inspected_region": int(region_id),
        "reveals_trace": bool(correct),
        "observed_trace_region": int(gt["true_trace_region"] if correct else gt["initial_trace_region"]),
        "epistemic_gain": 1.0 if correct and gt["needs_inspection"] else 0.0,
    }


def first_different_region(true_region: int, saliency_region: int, short_region: int, config: dict[str, Any]) -> int:
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    for candidate in [saliency_region, short_region, true_region + 1, true_region + 7]:
        region = int(candidate) % (grid_size * grid_size)
        if region != int(true_region):
            return region
    return (int(true_region) + 1) % (grid_size * grid_size)


def b5_costs(config: dict[str, Any]) -> dict[str, float]:
    costs = config.get("b5", {}).get("costs", {})
    return {
        "inspection_cost": float(costs.get("inspection_cost", 0.10)),
        "intervention_cost": float(costs.get("intervention_cost", 0.20)),
        "wrong_inspect_penalty": float(costs.get("wrong_inspect_penalty", 0.30)),
        "wrong_intervention_penalty": float(costs.get("wrong_intervention_penalty", 0.50)),
    }


def b5_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    base = resolve_b5_base_config(config)
    runtime = b42_runtime_config(base)
    b5 = b5_config(config)
    env = dict(runtime.get("env", {}))
    for key in ("frame_size", "grid_size", "past_frames", "future_frames"):
        if key in b5:
            env[key] = int(b5[key])
    runtime["env"] = env
    runtime["b5"] = b5
    return runtime


def resolve_b5_base_config(config: dict[str, Any]) -> dict[str, Any]:
    if "base_config" not in config:
        return config
    path = Path(str(config.get("base_config", "configs/b42_action_type_disambiguation.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def b5_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b5", {"n_test": 32, "n_ood": 32, "gates": {}})


def effective_count(value: int, b5: dict[str, Any]) -> int:
    cap = b5.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))
