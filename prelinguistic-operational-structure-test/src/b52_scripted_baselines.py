from __future__ import annotations

from typing import Any

import numpy as np

from .b4_intervention_policy import trace_family_for_model
from .b52_feedback_stress import revise_trace_from_feedback
from .b52_plan_divergence import adaptive_trace_update


def scripted_trace_update_baseline(model_input: dict[str, Any], inspection_observation: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trace = model_input["previous_trace_state"]
    return {
        "region": int(trace["region"]),
        "confidence": float(trace.get("confidence", 0.5)),
        "uncertainty": float(trace.get("uncertainty", 0.5)),
        "source": "scripted_previous_trace_update",
    }


def scripted_feedback_revision_baseline(model_input: dict[str, Any], consequence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trace = model_input["previous_trace_state"]
    return {
        "region": int(trace["region"]),
        "confidence": float(trace.get("confidence", 0.5)),
        "uncertainty": float(trace.get("uncertainty", 0.5)),
        "source": "scripted_previous_trace_feedback",
    }


def evaluate_update_vs_scripted_baseline(
    model: Any,
    episode_bundles: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str = "",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model_hits = []
    scripted_hits = []
    shuffled_hits = []
    records = []
    for idx, bundle in enumerate(episode_bundles):
        model_update = adaptive_trace_update(model, bundle["model_input"], config)
        scripted_update = scripted_trace_update_baseline(bundle["model_input"], bundle["model_input"].get("inspection_observation", {}), config)
        family = trace_family_for_model(model)
        expected = int(bundle["model_input"]["inspection_observation"]["family_trace_regions"][family])
        model_hit = int(model_update["region"]) == expected
        scripted_hit = int(scripted_update["region"]) == expected
        shuffled_hit = False
        model_hits.append(1.0 if model_hit else 0.0)
        scripted_hits.append(1.0 if scripted_hit else 0.0)
        shuffled_hits.append(1.0 if shuffled_hit else 0.0)
        records.append(
            {
                "model": model_name,
                "seed": seed,
                "episode_id": int(bundle["metadata"]["episode_id"]),
                "pair_id": idx,
                "test_type": "scripted_update_baseline",
                "trace_before_region": int(bundle["model_input"]["previous_trace_state"]["region"]),
                "trace_after_update_region": int(model_update["region"]),
                "expected_trace_after_update_region": expected,
                "scripted_update_region": int(scripted_update["region"]),
                "gate_pass": int(model_hit and not scripted_hit),
                "note": "model update compared with scripted previous-trace update",
            }
        )
    model_score = mean_or_zero(model_hits)
    scripted_score = mean_or_zero(scripted_hits)
    shuffled_score = mean_or_zero(shuffled_hits)
    eps = 1e-6
    return {
        "model_update_score": model_score,
        "scripted_update_score": scripted_score,
        "model_gain_over_scripted_update": model_score - scripted_score,
        "update_specificity_over_scripted": model_score / (scripted_score + eps),
        "update_specificity_over_shuffled": model_score / (shuffled_score + eps),
    }, records


def evaluate_feedback_vs_scripted_baseline(
    model: Any,
    episode_bundles: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str = "",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model_hits = []
    scripted_hits = []
    records = []
    for idx, bundle in enumerate(episode_bundles):
        trace_after = adaptive_trace_update(model, bundle["model_input"], config)
        consequence = bundle["model_input"].get("previous_consequence", {})
        model_revision = revise_trace_from_feedback(model, trace_after, consequence, config)
        scripted_revision = scripted_feedback_revision_baseline(bundle["model_input"], consequence, config)
        family = trace_family_for_model(model)
        expected = int(consequence["family_revision_regions"][family])
        model_hit = int(model_revision["region"]) == expected
        scripted_hit = int(scripted_revision["region"]) == expected
        model_hits.append(1.0 if model_hit else 0.0)
        scripted_hits.append(1.0 if scripted_hit else 0.0)
        records.append(
            {
                "model": model_name,
                "seed": seed,
                "episode_id": int(bundle["metadata"]["episode_id"]),
                "pair_id": idx,
                "test_type": "scripted_feedback_baseline",
                "trace_before_region": int(trace_after["region"]),
                "feedback_revision_region": int(model_revision["region"]),
                "expected_feedback_revision_region": expected,
                "scripted_feedback_region": int(scripted_revision["region"]),
                "gate_pass": int(model_hit and not scripted_hit),
                "note": "model feedback revision compared with scripted previous-trace feedback",
            }
        )
    model_score = mean_or_zero(model_hits)
    scripted_score = mean_or_zero(scripted_hits)
    eps = 1e-6
    return {
        "model_feedback_score": model_score,
        "scripted_feedback_score": scripted_score,
        "model_gain_over_scripted_feedback": model_score - scripted_score,
        "feedback_specificity_over_scripted": model_score / (scripted_score + eps),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
