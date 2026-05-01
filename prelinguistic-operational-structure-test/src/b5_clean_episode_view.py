from __future__ import annotations

import copy
from typing import Any

from .b5_closed_loop_env import b5_costs


FORBIDDEN_POLICY_KEYS = {
    "ground_truth",
    "evaluator_ground_truth",
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
    "best_region",
    "best_action",
    "best_plan",
    "value_table",
    "oracle_value",
    "expected_answer",
}

ALLOWED_MODEL_INPUT_KEYS = {
    "episode_id",
    "episode_type_public",
    "past_frames",
    "visible_state",
    "partial_observation",
    "occlusion_mask",
    "candidate_regions",
    "action_space",
    "cost_config_public",
    "previous_trace_state",
    "inspection_observation",
    "previous_consequence",
    "actionability_mask",
    "safe_metadata",
}


def split_b5_episode_for_clean_run(raw_episode: dict[str, Any], config: dict[str, Any], episode_id: int = 0) -> dict[str, Any]:
    model_input = make_sanitized_model_input(raw_episode, config, episode_id)
    evaluator_ground_truth = make_evaluator_ground_truth(raw_episode, config)
    oracle_baseline_view = make_oracle_baseline_view(raw_episode, config)
    assert_model_input_is_sanitized(model_input, config)
    return {
        "model_input": model_input,
        "evaluator_ground_truth": evaluator_ground_truth,
        "oracle_baseline_view": oracle_baseline_view,
        "metadata": {
            "episode_id": int(episode_id),
            "episode_type": evaluator_ground_truth.get("closed_loop_episode_type", ""),
            "family": evaluator_ground_truth.get("family", ""),
        },
    }


def make_sanitized_model_input(raw_episode: dict[str, Any], config: dict[str, Any], episode_id: int = 0) -> dict[str, Any]:
    gt = raw_episode.get("ground_truth", {})
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    trace_state = copy.deepcopy(raw_episode.get("trace_state", {}))
    epistemic_scores = raw_episode.get("closed_loop_signal", {}).get("epistemic_scores", {})
    action_scores = raw_episode.get("closed_loop_signal", {}).get("action_type_signal", raw_episode.get("action_type_signal", {}))
    model_input = {
        "episode_id": int(episode_id),
        "episode_type_public": public_episode_type(str(gt.get("closed_loop_episode_type", ""))),
        "past_frames": copy.deepcopy(raw_episode.get("past_frames", [])),
        "visible_state": {
            "saliency_region": int(gt.get("saliency_region", 0)),
            "short_horizon_region": int(gt.get("short_horizon_region", 0)),
        },
        "partial_observation": copy.deepcopy(raw_episode.get("partial_observation", {})),
        "occlusion_mask": copy.deepcopy(raw_episode.get("occlusion_mask", [])),
        "candidate_regions": [
            {
                "region_id": region,
                "public_trace_score": float(epistemic_scores.get(str(region), epistemic_scores.get(region, 0.0))),
            }
            for region in range(grid_size * grid_size)
        ],
        "action_space": [
            {"action_type": str(action_type), "public_score": float(score)}
            for action_type, score in sorted(action_scores.items())
        ],
        "cost_config_public": b5_costs(config),
        "previous_trace_state": {
            "family": trace_state.get("family", gt.get("family", "")),
            "region": int(trace_state.get("region", gt.get("initial_trace_region", 0))),
            "confidence": float(trace_state.get("confidence", 0.5)),
            "uncertainty": float(trace_state.get("uncertainty", 0.5)),
            "source": "sanitized_previous_trace",
        },
        "inspection_observation": copy.deepcopy(raw_episode.get("inspection_observation", {})),
        "previous_consequence": copy.deepcopy(raw_episode.get("previous_consequence", {})),
        "safe_metadata": {
            "family": gt.get("family", ""),
            "delay": int(gt.get("delay", 0)),
        },
    }
    return model_input


def make_evaluator_ground_truth(raw_episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gt = copy.deepcopy(raw_episode.get("ground_truth", {}))
    return {
        "closed_loop_episode_type": gt.get("closed_loop_episode_type", ""),
        "family": gt.get("family", ""),
        "needs_inspection": bool(gt.get("needs_inspection", False)),
        "should_do_nothing": bool(gt.get("should_do_nothing", False)),
        "true_trace_region": int(gt.get("true_trace_region", 0)),
        "initial_trace_region": int(gt.get("initial_trace_region", 0)),
        "oracle_inspect_region": int(gt.get("oracle_inspect_region", gt.get("true_trace_region", 0))),
        "wrong_inspect_region": int(gt.get("wrong_inspect_region", gt.get("initial_trace_region", 0))),
        "trace_after_inspection_region": int(gt.get("trace_after_inspection_region", gt.get("true_trace_region", 0))),
        "oracle_intervention_action": copy.deepcopy(gt.get("oracle_intervention_action", {"action_type": "do_nothing", "region_id": 0, "strength": 0.0})),
        "oracle_closed_loop_plan": copy.deepcopy(gt.get("oracle_closed_loop_plan", {})),
        "inspection_cost": float(gt.get("inspection_cost", b5_costs(config)["inspection_cost"])),
        "intervention_cost": float(gt.get("intervention_cost", b5_costs(config)["intervention_cost"])),
    }


def make_oracle_baseline_view(raw_episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gt = raw_episode.get("ground_truth", {})
    return {
        "oracle_closed_loop_plan": copy.deepcopy(gt.get("oracle_closed_loop_plan", {})),
        "oracle_inspect_region": int(gt.get("oracle_inspect_region", gt.get("true_trace_region", 0))),
        "oracle_intervention_action": copy.deepcopy(gt.get("oracle_intervention_action", {})),
    }


def assert_model_input_is_sanitized(model_input: dict[str, Any], config: dict[str, Any]) -> None:
    forbidden = set(config.get("b5_clean", {}).get("forbidden_policy_keys", FORBIDDEN_POLICY_KEYS))
    paths = find_forbidden_key_paths(model_input, forbidden)
    disallowed_top = [key for key in model_input if key not in ALLOWED_MODEL_INPUT_KEYS]
    if paths or disallowed_top:
        raise AssertionError(f"Unsanitized model_input: paths={paths}, disallowed_top={disallowed_top}")


def find_forbidden_key_paths(obj: Any, forbidden_keys: set[str], prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in forbidden_keys:
                paths.append(key_path)
            paths.extend(find_forbidden_key_paths(value, forbidden_keys, key_path))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            paths.extend(find_forbidden_key_paths(value, forbidden_keys, f"{prefix}[{idx}]"))
    return paths


def public_episode_type(value: str) -> str:
    if value in {"inspect_needed", "intervention_clear", "no_action", "misleading_initial_trace"}:
        return value
    return "unknown"
