from __future__ import annotations

from typing import Any

import numpy as np

from .b32_inspection_values import best_region
from .b32_mechanism_policy import cached_family_scores
from .b4_intervention_policy import MODEL_TO_TRACE_FAMILY, trace_family_for_model
from .b42_action_type_env import allowed_action_types
from .b42_action_type_metrics import action_type_accuracy, joint_region_action_accuracy, region_accuracy


def action_type_disambiguating_policy(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trace_family = trace_family_for_model(model)
    family_scores = cached_family_scores(model, episode, config)
    region_scores = dict(family_scores.get(trace_family, {}))
    region = best_region(region_scores)
    action_type_scores = score_action_types(model, episode, config)
    action_type = max(action_type_scores.items(), key=lambda item: (float(item[1]), item[0]))[0]
    action = {
        "action_type": action_type,
        "region_id": int(region),
        "strength": float(config.get("b42", config.get("b4", {})).get("action_strength", 1.0)),
    }
    candidate_scores = {
        f"{candidate}:{region}": float(score * max(region_scores.get(region, 0.0), 0.0))
        for candidate, score in action_type_scores.items()
    }
    return {
        "action": action,
        "trace_family": trace_family,
        "policy_source": f"{trace_family}_action_type_disambiguating_policy",
        "region_score": float(region_scores.get(region, 0.0)),
        "action_type_score": float(action_type_scores[action_type]),
        "action_type_scores": action_type_scores,
        "candidate_action_scores": candidate_scores,
        "provenance": {
            "private_trace_used": True,
            "oracle_value_used": False,
            "fixed_action_policy_used": bool(episode.get("b42_action_type_ablation")),
            "shared_action_policy_used": False,
        },
    }


def score_action_types(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    allowed = allowed_action_types(config)
    if episode.get("b42_action_type_ablation") or episode.get("b42_trace_action_ablation"):
        fallback = str(episode.get("b42_ablation_fallback_action", allowed[0]))
        return {action_type: (1.0 if action_type == fallback else 0.0) for action_type in allowed if action_type != episode.get("b42_masked_action_type")}
    signal = {action_type: float(episode.get("action_type_signal", {}).get(action_type, 0.0)) for action_type in allowed}
    masked = episode.get("b42_masked_action_type")
    if masked in signal:
        signal.pop(masked)
    return signal


def evaluate_action_type_policy(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records = []
    action_hits = []
    region_hits = []
    joint_hits = []
    for episode_id, episode in enumerate(episodes):
        policy = action_type_disambiguating_policy(model, episode, config)
        action = policy["action"]
        expected = episode["ground_truth"]["oracle_best_action"]
        action_hit = action_type_accuracy(action, expected)
        region_hit = region_accuracy(action, expected)
        joint_hit = joint_region_action_accuracy(action, expected)
        action_hits.append(action_hit)
        region_hits.append(region_hit)
        joint_hits.append(joint_hit)
        records.append(
            {
                "record_kind": "policy",
                "model": model_name or str(getattr(model, "name", "")),
                "seed": seed,
                "episode_id": episode_id,
                "family": episode["ground_truth"].get("family", ""),
                "trace_family": policy["trace_family"],
                "true_trace_region": int(episode["ground_truth"]["true_trace_region"]),
                "predicted_region": int(action["region_id"]),
                "expected_region": int(expected["region_id"]),
                "predicted_action_type": action["action_type"],
                "expected_action_type": expected["action_type"],
                "required_action_type": episode["ground_truth"]["required_action_type"],
                "gate_pass": int(joint_hit),
                "note": "action-type-disambiguating private trace policy",
            }
        )
    return {
        "action_type_accuracy": mean_or_zero(action_hits),
        "region_accuracy": mean_or_zero(region_hits),
        "joint_region_action_accuracy": mean_or_zero(joint_hits),
    }, records


def evaluate_family_action_mapping(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    metrics, records = evaluate_action_type_policy(model, episodes, config, seed, model_name)
    by_family: dict[str, list[float]] = {"recurrent": [], "field": [], "schema": []}
    action_types = []
    for row in records:
        by_family.setdefault(str(row["family"]), []).append(float(row["gate_pass"]))
        action_types.append(str(row["predicted_action_type"]))
        row["record_kind"] = "family_action_mapping"
    metrics.update(
        {
            "family_action_mapping_accuracy": metrics["joint_region_action_accuracy"],
            "recurrent_action_accuracy": mean_or_zero(by_family.get("recurrent", [])),
            "field_action_accuracy": mean_or_zero(by_family.get("field", [])),
            "schema_action_accuracy": mean_or_zero(by_family.get("schema", [])),
            "family_action_diversity": len(set(action_types)) / max(1, len(allowed_action_types(config))),
        }
    )
    return metrics, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

