from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b5_clean_episode_view import assert_model_input_is_sanitized
from .b52_plan_divergence import adaptive_intervention_plan, adaptive_trace_update, closed_loop_plan_signature_after_update


def make_counterfactual_inspection_observation(
    episode_bundle: dict[str, Any],
    counterfactual_type: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    obs = copy.deepcopy(episode_bundle["model_input"].get("inspection_observation", {}))
    regions = dict(obs.get("family_trace_regions", {}))
    shift = {"trace_shifted": 3, "delayed_influence_absent": 11, "alternative_occluded_path": 17}.get(counterfactual_type, 5)
    total = int(config.get("env", {}).get("grid_size", 8)) ** 2
    obs["content_id"] = f"{obs.get('content_id', 'obs')}_{counterfactual_type}"
    obs["counterfactual_type"] = counterfactual_type
    obs["family_trace_regions"] = {family: (int(region) + shift) % total for family, region in regions.items()}
    return obs


def evaluate_counterfactual_inspection_update(
    model: Any,
    episode_bundles: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str = "",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    update_switches = []
    plan_switches = []
    intervention_switches = []
    alignments = []
    records = []
    for episode_id, bundle in enumerate(episode_bundles):
        base_output = adaptive_intervention_plan(model, bundle["model_input"], config)
        base_update = base_output["trace_after_update"]
        base_plan = closed_loop_plan_signature_after_update(base_output)
        for cf_type in ["trace_shifted", "delayed_influence_absent", "alternative_occluded_path"]:
            cf_bundle = copy.deepcopy(bundle)
            cf_bundle["model_input"]["inspection_observation"] = make_counterfactual_inspection_observation(bundle, cf_type, config)
            assert_model_input_is_sanitized(cf_bundle["model_input"], config)
            cf_output = adaptive_intervention_plan(model, cf_bundle["model_input"], config)
            cf_update = adaptive_trace_update(model, cf_bundle["model_input"], config)
            family = cf_output["trace_family"]
            expected = int(cf_bundle["model_input"]["inspection_observation"]["family_trace_regions"][family])
            update_switched = int(base_update["region"]) != int(cf_update["region"])
            plan_switched = base_plan != closed_loop_plan_signature_after_update(cf_output)
            intervention_switched = base_output["intervention_action"] != cf_output["intervention_action"]
            aligned = int(cf_update["region"]) == expected
            update_switches.append(1.0 if update_switched else 0.0)
            plan_switches.append(1.0 if plan_switched else 0.0)
            intervention_switches.append(1.0 if intervention_switched else 0.0)
            alignments.append(1.0 if aligned else 0.0)
            records.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "episode_id": int(bundle["metadata"]["episode_id"]),
                    "pair_id": episode_id,
                    "test_type": "counterfactual_inspection",
                    "counterfactual_type": cf_type,
                    "trace_before_region": int(bundle["model_input"]["previous_trace_state"]["region"]),
                    "trace_after_update_region": int(cf_update["region"]),
                    "expected_trace_after_update_region": expected,
                    "intervention_before_update": f'{base_output["intervention_action"]["action_type"]}:{base_output["intervention_action"]["region_id"]}',
                    "intervention_after_update": f'{cf_output["intervention_action"]["action_type"]}:{cf_output["intervention_action"]["region_id"]}',
                    "plan_signature_before": str(base_plan),
                    "plan_signature_after": str(closed_loop_plan_signature_after_update(cf_output)),
                    "gate_pass": int(update_switched and plan_switched and aligned),
                    "note": "counterfactual inspection changes update and plan",
                }
            )
    return {
        "counterfactual_update_switch_rate": mean_or_zero(update_switches),
        "counterfactual_plan_switch_rate": mean_or_zero(plan_switches),
        "counterfactual_intervention_switch_rate": mean_or_zero(intervention_switches),
        "counterfactual_update_alignment": mean_or_zero(alignments),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
