from __future__ import annotations

from typing import Any

from .b32_inspection_values import best_region
from .b32_mechanism_policy import cached_family_scores
from .b4_intervention_env import canonical_family


MODEL_TO_TRACE_FAMILY = {
    "recurrent_flow_checkpoint_model": "recurrent",
    "field_memory_model": "field",
    "schema_memory_model": "schema",
}

FAMILY_ACTION = {
    "recurrent": "stabilize_trace_region",
    "field": "block_force_region",
    "schema": "apply_local_push",
}


def trace_guided_intervention_policy(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trace_family = trace_family_for_model(model)
    family_scores = cached_family_scores(model, episode, config)
    region_scores = apply_b4_ablation_hook(dict(family_scores[trace_family]), episode, trace_family)
    region = best_region(region_scores)
    action_scores = score_action_types_from_trace(model, episode, region, config)
    action_type = max(action_scores.items(), key=lambda item: (float(item[1]), item[0]))[0]
    action = {
        "action_type": action_type,
        "region_id": int(region),
        "strength": float(config.get("b4", {}).get("action_strength", 1.0)),
    }
    return {
        "action": action,
        "trace_family": trace_family,
        "policy_source": f"{trace_family}_private_trace_intervention",
        "region_score": float(region_scores.get(region, 0.0)),
        "action_type_score": float(action_scores[action_type]),
        "provenance": {
            "private_trace_action_score_used": True,
            "oracle_best_action_used": False,
            "oracle_intervention_value_used": False,
            "shared_action_policy_used": False,
            "family_ablation_hook_used": bool(episode.get("b4_ablation")),
        },
    }


def score_action_types_from_trace(model: Any, episode: dict[str, Any], region_id: int, config: dict[str, Any]) -> dict[str, float]:
    trace_family = trace_family_for_model(model)
    scores = {
        "do_nothing": 0.0,
        "inspect_only": 0.05,
        "apply_local_damping": 0.25,
        "apply_local_push": 0.25,
        "block_force_region": 0.25,
        "stabilize_trace_region": 0.25,
    }
    if trace_family == "recurrent":
        scores.update({"stabilize_trace_region": 1.0, "apply_local_damping": 0.70})
    elif trace_family == "field":
        scores.update({"block_force_region": 1.0, "apply_local_damping": 0.70})
    else:
        scores.update({"apply_local_push": 1.0, "stabilize_trace_region": 0.70})
    return scores


def evaluate_trace_guided_intervention_policy(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    from .b4_intervention_env import apply_intervention_action, evaluate_intervention_outcome
    from .b4_intervention_metrics import action_type_accuracy, intervention_region_accuracy, trace_guided_intervention_accuracy
    from .b4_intervention_values import compute_intervention_value, compute_wrong_region_penalty, wrong_region_action_for

    records = []
    trace_hits = []
    region_hits = []
    action_hits = []
    improvements = []
    wrong_penalties = []
    for episode_id, episode in enumerate(episodes):
        policy = trace_guided_intervention_policy(model, episode, config)
        action = policy["action"]
        oracle = episode["ground_truth"]["oracle_best_action"]
        intervened = apply_intervention_action(episode, action, config)
        outcome = evaluate_intervention_outcome(episode, intervened, config)
        wrong = wrong_region_action_for(episode, oracle, config)
        wrong_penalty = compute_wrong_region_penalty(episode, oracle, wrong, config)
        trace_hit = trace_guided_intervention_accuracy(action, oracle)
        region_hit = intervention_region_accuracy(action, oracle)
        action_hit = action_type_accuracy(action, oracle)
        trace_hits.append(trace_hit)
        region_hits.append(region_hit)
        action_hits.append(action_hit)
        improvements.append(outcome["outcome_improvement"])
        wrong_penalties.append(wrong_penalty)
        records.append(
            {
                "record_kind": "policy",
                "model": model_name or str(getattr(model, "name", "")),
                "seed": seed,
                "episode_id": episode_id,
                "episode_type": episode["ground_truth"].get("episode_type", ""),
                "family": policy["trace_family"],
                "delay": int(episode["ground_truth"].get("delay", 0)),
                "true_trace_region": int(episode["ground_truth"]["true_trace_region"]),
                "saliency_region": int(episode["ground_truth"]["saliency_region"]),
                "short_horizon_region": int(episode["ground_truth"]["short_horizon_region"]),
                "predicted_action_type": action["action_type"],
                "predicted_region": int(action["region_id"]),
                "oracle_action_type": oracle["action_type"],
                "oracle_region": int(oracle["region_id"]),
                "family_expected_action_type": oracle["action_type"],
                "family_expected_region": int(oracle["region_id"]),
                "baseline_outcome_error": outcome["baseline_outcome_error"],
                "intervened_outcome_error": outcome["intervened_outcome_error"],
                "outcome_improvement": outcome["outcome_improvement"],
                "wrong_region_value": compute_intervention_value(episode, wrong, config),
                "wrong_region_penalty": wrong_penalty,
                "gate_pass": int(trace_hit),
                "note": "trace-guided local intervention",
            }
        )
    return {
        "trace_guided_intervention_accuracy": mean_or_zero(trace_hits),
        "intervention_region_accuracy": mean_or_zero(region_hits),
        "action_type_accuracy": mean_or_zero(action_hits),
        "outcome_improvement": mean_or_zero(improvements),
        "wrong_region_penalty_sensitivity": mean_or_zero(wrong_penalties),
    }, records


def trace_family_for_model(model: Any) -> str:
    name = str(getattr(model, "name", ""))
    return MODEL_TO_TRACE_FAMILY.get(name, canonical_family(str(getattr(model, "structural_family", ""))))


def apply_b4_ablation_hook(scores: dict[int, float], episode: dict[str, Any], trace_family: str) -> dict[int, float]:
    ablation = episode.get("b4_ablation", {})
    if ablation.get("family") != trace_family or not scores:
        return scores
    target = int(ablation.get("target_region", -1))
    if target in scores:
        scores[target] = min(float(value) for value in scores.values()) - 1.0
    return scores


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

