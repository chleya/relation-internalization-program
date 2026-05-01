from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .b4_intervention_policy import trace_family_for_model
from .b52_adaptive_update_env import inspection_content_signature, observation_signature


def adaptive_trace_update(model: Any, model_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trace_before = dict(model_input["previous_trace_state"])
    obs = model_input.get("inspection_observation", {})
    family = trace_family_for_model(model)
    regions = obs.get("family_trace_regions", {})
    if family in regions:
        return {
            "region": int(regions[family]),
            "confidence": 0.95,
            "uncertainty": 0.05,
            "source": "content_conditioned_private_trace_update",
            "trace_family": family,
        }
    if obs.get("reveals_trace", False):
        return {
            "region": int(obs.get("observed_trace_region", trace_before.get("region", 0))),
            "confidence": 0.85,
            "uncertainty": 0.15,
            "source": "content_conditioned_trace_update",
            "trace_family": family,
        }
    return trace_before


def adaptive_intervention_plan(model: Any, model_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    family = trace_family_for_model(model)
    trace_after = adaptive_trace_update(model, model_input, config)
    action_types = model_input.get("inspection_observation", {}).get("family_action_types", {})
    action_type = str(action_types.get(family, "apply_local_damping"))
    action = {"action_type": action_type, "region_id": int(trace_after["region"]), "strength": 1.0}
    return {
        "trace_family": family,
        "trace_before": dict(model_input["previous_trace_state"]),
        "trace_after_update": trace_after,
        "intervention_action": action,
        "policy_source": "b52_content_conditioned_private_trace_update",
        "provenance": {
            "private_trace_used": True,
            "oracle_plan_used": False,
            "oracle_trace_update_used": False,
            "oracle_feedback_revision_used": False,
            "oracle_value_used": False,
        },
    }


def closed_loop_plan_signature_after_update(policy_output: dict[str, Any]) -> tuple:
    action = policy_output["intervention_action"]
    trace_after = policy_output["trace_after_update"]
    return (
        int(trace_after["region"]),
        action["action_type"],
        int(action["region_id"]),
        policy_output.get("trace_family", ""),
    )


def evaluate_same_initial_different_info_plan_divergence(
    model: Any,
    episode_pairs: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str = "",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    divergence = []
    plan_divergence = []
    intervention_change = []
    records = []
    for pair_id, pair in enumerate(episode_pairs):
        bundle_a = pair["episode_a"]
        bundle_b = pair["episode_b"]
        out_a = adaptive_intervention_plan(model, bundle_a["model_input"], config)
        out_b = adaptive_intervention_plan(model, bundle_b["model_input"], config)
        sig_a = closed_loop_plan_signature_after_update(out_a)
        sig_b = closed_loop_plan_signature_after_update(out_b)
        different_plan = sig_a != sig_b
        different_action = out_a["intervention_action"] != out_b["intervention_action"]
        divergence.append(1.0 if different_plan else 0.0)
        plan_divergence.append(1.0 if different_plan else 0.0)
        intervention_change.append(1.0 if different_action else 0.0)
        for suffix, bundle, output, other_sig in [("a", bundle_a, out_a, sig_b), ("b", bundle_b, out_b, sig_a)]:
            records.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "episode_id": int(bundle["metadata"]["episode_id"]),
                    "pair_id": pair_id,
                    "test_type": "plan_divergence",
                    "initial_observation_signature": str(observation_signature(bundle)),
                    "inspection_content_signature": str(inspection_content_signature(bundle)),
                    "trace_before_region": int(output["trace_before"]["region"]),
                    "trace_after_update_region": int(output["trace_after_update"]["region"]),
                    "expected_trace_after_update_region": int(bundle["model_input"]["inspection_observation"]["family_trace_regions"][output["trace_family"]]),
                    "intervention_before_update": "none",
                    "intervention_after_update": f'{output["intervention_action"]["action_type"]}:{output["intervention_action"]["region_id"]}',
                    "plan_signature_before": str(other_sig),
                    "plan_signature_after": str(closed_loop_plan_signature_after_update(output)),
                    "gate_pass": int(different_plan),
                    "note": f"same initial different info plan divergence {suffix}",
                }
            )
    return {
        "same_initial_different_info_plan_divergence": mean_or_zero(divergence),
        "post_update_plan_divergence": mean_or_zero(plan_divergence),
        "post_update_intervention_change_rate": mean_or_zero(intervention_change),
    }, records


def compute_cross_model_plan_overlap(records: list[dict[str, Any]]) -> dict[str, float]:
    by_episode: dict[int, list[str]] = {}
    for record in records:
        if record.get("test_type") != "plan_divergence":
            continue
        by_episode.setdefault(int(record.get("episode_id", 0)), []).append(str(record.get("plan_signature_after", "")))
    pairwise = []
    exact_all = []
    for signatures in by_episode.values():
        if len(signatures) < 2:
            continue
        exact_all.append(1.0 if len(set(signatures)) == 1 else 0.0)
        pairwise.extend(1.0 if a == b else 0.0 for a, b in combinations(signatures, 2))
    return {
        "cross_model_exact_plan_match_rate": mean_or_zero(pairwise),
        "exact_all_model_same_plan_rate": mean_or_zero(exact_all),
    }


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
