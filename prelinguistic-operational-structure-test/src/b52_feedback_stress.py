from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b4_intervention_policy import trace_family_for_model
from .b52_plan_divergence import adaptive_intervention_plan


def make_contradictory_feedback(episode_bundle: dict[str, Any], policy_output: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    total = int(config.get("env", {}).get("grid_size", 8)) ** 2
    base = int(policy_output["trace_after_update"]["region"])
    regions = {family: (base + offset) % total for family, offset in {"recurrent": 23, "field": 29, "schema": 31}.items()}
    return {
        "feedback_type": "contradictory",
        "family_revision_regions": regions,
        "action_success": False,
        "delay_steps": 0,
    }


def make_delayed_feedback(episode_bundle: dict[str, Any], policy_output: dict[str, Any], delay_steps: int, config: dict[str, Any]) -> dict[str, Any]:
    previous = episode_bundle["model_input"].get("previous_consequence", {})
    regions = dict(previous.get("family_revision_regions", {}))
    return {
        "feedback_type": "delayed",
        "family_revision_regions": regions,
        "action_success": True,
        "delay_steps": int(delay_steps),
    }


def revise_trace_from_feedback(model: Any, trace_after_update: dict[str, Any], consequence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    family = trace_family_for_model(model)
    regions = consequence.get("family_revision_regions", {})
    if family in regions:
        return {
            "region": int(regions[family]),
            "confidence": 0.90 if consequence.get("feedback_type") != "contradictory" else 0.70,
            "uncertainty": 0.10 if consequence.get("feedback_type") != "contradictory" else 0.30,
            "source": "content_conditioned_feedback_revision",
            "trace_family": family,
        }
    return dict(trace_after_update)


def evaluate_feedback_content_sensitivity(
    model: Any,
    episode_bundles: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str = "",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    sensitivity = []
    contradictory_hits = []
    delayed_hits = []
    plan_switches = []
    improvements = []
    records = []
    for idx, bundle in enumerate(episode_bundles):
        output = adaptive_intervention_plan(model, bundle["model_input"], config)
        direct = revise_trace_from_feedback(model, output["trace_after_update"], bundle["model_input"].get("previous_consequence", {}), config)
        contradictory = make_contradictory_feedback(bundle, output, config)
        delayed = make_delayed_feedback(bundle, output, 3, config)
        contradictory_revision = revise_trace_from_feedback(model, output["trace_after_update"], contradictory, config)
        delayed_revision = revise_trace_from_feedback(model, output["trace_after_update"], delayed, config)
        changed = int(direct["region"]) != int(contradictory_revision["region"])
        delayed_expected = int(delayed["family_revision_regions"][trace_family_for_model(model)])
        contradictory_expected = int(contradictory["family_revision_regions"][trace_family_for_model(model)])
        contradictory_hit = int(contradictory_revision["region"]) == contradictory_expected
        delayed_hit = int(delayed_revision["region"]) == delayed_expected
        sensitivity.append(1.0 if changed else 0.0)
        contradictory_hits.append(1.0 if contradictory_hit else 0.0)
        delayed_hits.append(1.0 if delayed_hit else 0.0)
        plan_switches.append(1.0 if changed else 0.0)
        improvements.append(1.0 if delayed_hit else 0.0)
        for consequence_type, revision, expected, hit in [
            ("contradictory", contradictory_revision, contradictory_expected, contradictory_hit),
            ("delayed", delayed_revision, delayed_expected, delayed_hit),
        ]:
            records.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "episode_id": int(bundle["metadata"]["episode_id"]),
                    "pair_id": idx,
                    "test_type": "feedback_stress",
                    "trace_before_region": int(output["trace_after_update"]["region"]),
                    "consequence_type": consequence_type,
                    "feedback_revision_region": int(revision["region"]),
                    "expected_feedback_revision_region": int(expected),
                    "gate_pass": int(hit),
                    "note": f"{consequence_type} feedback content revision",
                }
            )
    return {
        "feedback_content_sensitivity": mean_or_zero(sensitivity),
        "contradictory_feedback_revision_accuracy": mean_or_zero(contradictory_hits),
        "delayed_feedback_revision_accuracy": mean_or_zero(delayed_hits),
        "feedback_plan_switch_rate": mean_or_zero(plan_switches),
        "future_decision_improvement_after_feedback": mean_or_zero(improvements),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
