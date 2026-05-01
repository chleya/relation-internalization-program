from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .b5_clean_episode_view import assert_model_input_is_sanitized, split_b5_episode_for_clean_run
from .b5_clean_runner import b5_clean_runtime_config
from .b5_closed_loop_env import EPISODE_TYPES, make_b5_closed_loop_episode


FAMILIES = ["recurrent", "field", "schema"]
ACTION_BY_FAMILY = {
    "recurrent": "apply_local_damping",
    "field": "block_force_region",
    "schema": "stabilize_trace_region",
}


def make_b52_adaptive_update_episode_pair(config: dict[str, Any], seed: int, pair_type: str) -> dict[str, Any]:
    runtime = b52_runtime_config(config)
    pair_type = str(pair_type)
    raw_a = make_b5_closed_loop_episode(runtime, seed, EPISODE_TYPES[seed % len(EPISODE_TYPES)], FAMILIES[seed % 3])
    raw_b = make_b5_closed_loop_episode(runtime, seed + 7919, EPISODE_TYPES[(seed + 1) % len(EPISODE_TYPES)], FAMILIES[(seed + 1) % 3])
    bundle_a = split_b5_episode_for_clean_run(raw_a, runtime, seed)
    bundle_b = split_b5_episode_for_clean_run(raw_b, runtime, seed + 100000)
    make_initial_views_match(bundle_a, bundle_b)
    attach_adaptive_content(bundle_a, seed, variant=0, pair_type=pair_type)
    attach_adaptive_content(bundle_b, seed, variant=1, pair_type=pair_type)
    assert_model_input_is_sanitized(bundle_a["model_input"], runtime)
    assert_model_input_is_sanitized(bundle_b["model_input"], runtime)
    return {
        "episode_a": bundle_a,
        "episode_b": bundle_b,
        "pair_type": pair_type,
        "expected_difference": {
            "trace_update_should_differ": True,
            "intervention_plan_should_differ": True,
            "feedback_revision_should_differ": True,
        },
        "evaluator_ground_truth_pair": {
            "expected_a_regions": bundle_a["model_input"]["inspection_observation"]["family_trace_regions"],
            "expected_b_regions": bundle_b["model_input"]["inspection_observation"]["family_trace_regions"],
        },
    }


def make_b52_datasets(config: dict[str, Any], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    b52 = b52_config(config)
    n_pairs = effective_count(int(b52.get("n_stress_pairs", 32)), b52)
    pair_types = [
        "same_initial_different_inspection",
        "inspection_content_swap",
        "counterfactual_inspection",
        "contradictory_feedback",
        "delayed_feedback",
        "scripted_update_trap",
        "scripted_feedback_trap",
    ]
    pairs = [make_b52_adaptive_update_episode_pair(config, seed + idx * 29, pair_types[idx % len(pair_types)]) for idx in range(n_pairs)]
    return {"pairs": pairs, "bundles": [bundle for pair in pairs for bundle in (pair["episode_a"], pair["episode_b"])]}


def make_initial_views_match(bundle_a: dict[str, Any], bundle_b: dict[str, Any]) -> None:
    source = bundle_a["model_input"]
    target = bundle_b["model_input"]
    for key in ["episode_type_public", "visible_state", "partial_observation", "occlusion_mask", "candidate_regions", "action_space", "cost_config_public", "previous_trace_state", "previous_consequence"]:
        target[key] = copy.deepcopy(source[key])
    target["inspection_observation"] = {}
    source["inspection_observation"] = {}


def attach_adaptive_content(bundle: dict[str, Any], seed: int, variant: int, pair_type: str) -> None:
    grid_size = 8
    if "env" in bundle:
        grid_size = int(bundle["env"].get("grid_size", 8))
    total = grid_size * grid_size
    base = (seed * 7 + variant * 11) % total
    regions = {
        "recurrent": base % total,
        "field": (base + 9 + variant) % total,
        "schema": (base + 19 + 2 * variant) % total,
    }
    revision_regions = {
        "recurrent": (regions["recurrent"] + 5 + variant) % total,
        "field": (regions["field"] + 7 + variant) % total,
        "schema": (regions["schema"] + 13 + variant) % total,
    }
    bundle["model_input"]["inspection_observation"] = {
        "content_id": f"{pair_type}_{seed}_{variant}",
        "reveals_trace": True,
        "family_trace_regions": regions,
        "family_action_types": dict(ACTION_BY_FAMILY),
        "observed_trace_region": regions["recurrent"],
        "content_source": "sanitized_inspection_observation",
    }
    bundle["model_input"]["previous_consequence"] = {
        "consequence_id": f"{pair_type}_{seed}_{variant}",
        "feedback_type": "delayed" if pair_type == "delayed_feedback" else "contradictory" if pair_type == "contradictory_feedback" else "direct",
        "family_revision_regions": revision_regions,
        "delay_steps": 3 if pair_type == "delayed_feedback" else 0,
        "content_source": "sanitized_consequence_feedback",
    }
    bundle["evaluator_ground_truth"]["b52_expected_update_regions"] = dict(regions)
    bundle["evaluator_ground_truth"]["b52_expected_feedback_regions"] = dict(revision_regions)


def observation_signature(bundle: dict[str, Any]) -> tuple:
    model_input = bundle["model_input"]
    trace = model_input["previous_trace_state"]
    return (
        model_input.get("episode_type_public", ""),
        int(trace.get("region", -1)),
        round(float(trace.get("uncertainty", 0.0)), 3),
        tuple(sorted((int(row["region_id"]), round(float(row.get("public_trace_score", 0.0)), 3)) for row in model_input.get("candidate_regions", [])[:8])),
    )


def inspection_content_signature(bundle: dict[str, Any]) -> tuple:
    obs = bundle["model_input"].get("inspection_observation", {})
    return (
        obs.get("content_id", ""),
        tuple(sorted((family, int(region)) for family, region in obs.get("family_trace_regions", {}).items())),
        tuple(sorted((family, action) for family, action in obs.get("family_action_types", {}).items())),
    )


def b52_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    if "base_config" not in config:
        return b5_clean_runtime_config(config)
    path = Path(str(config.get("base_config", "configs/b5_clean_closed_loop.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    runtime = b5_clean_runtime_config(base)
    runtime["b52"] = b52_config(config)
    return runtime


def b52_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b52", {"n_stress_pairs": 32, "gates": {}})


def effective_count(value: int, b52: dict[str, Any]) -> int:
    cap = b52.get("max_eval_pairs")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))
