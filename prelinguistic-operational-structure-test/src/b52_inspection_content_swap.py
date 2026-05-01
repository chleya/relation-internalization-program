from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b5_clean_episode_view import assert_model_input_is_sanitized
from .b52_plan_divergence import adaptive_trace_update


def swap_inspection_content(
    episode_bundle_a: dict[str, Any],
    episode_bundle_b: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    swapped_a = copy.deepcopy(episode_bundle_a)
    swapped_b = copy.deepcopy(episode_bundle_b)
    obs_a = copy.deepcopy(episode_bundle_a["model_input"].get("inspection_observation", {}))
    obs_b = copy.deepcopy(episode_bundle_b["model_input"].get("inspection_observation", {}))
    swapped_a["model_input"]["inspection_observation"] = obs_b
    swapped_b["model_input"]["inspection_observation"] = obs_a
    assert_model_input_is_sanitized(swapped_a["model_input"], config)
    assert_model_input_is_sanitized(swapped_b["model_input"], config)
    return swapped_a, swapped_b


def evaluate_inspection_content_swap(
    model: Any,
    episode_pairs: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str = "",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    sensitivities = []
    change_rates = []
    alignments = []
    records = []
    for pair_id, pair in enumerate(episode_pairs):
        a = pair["episode_a"]
        b = pair["episode_b"]
        swapped_a, swapped_b = swap_inspection_content(a, b, config)
        for original, swapped, label in [(a, swapped_a, "a"), (b, swapped_b, "b")]:
            base_update = adaptive_trace_update(model, original["model_input"], config)
            swapped_update = adaptive_trace_update(model, swapped["model_input"], config)
            family = swapped_update.get("trace_family", "")
            expected = int(swapped["model_input"]["inspection_observation"]["family_trace_regions"][family])
            changed = int(base_update["region"]) != int(swapped_update["region"])
            aligned = int(swapped_update["region"]) == expected
            sensitivities.append(1.0 if changed and aligned else 0.0)
            change_rates.append(1.0 if changed else 0.0)
            alignments.append(1.0 if aligned else 0.0)
            records.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "episode_id": int(original["metadata"]["episode_id"]),
                    "pair_id": pair_id,
                    "test_type": "inspection_content_swap",
                    "inspection_content_signature": str(swapped["model_input"]["inspection_observation"].get("content_id", "")),
                    "trace_before_region": int(original["model_input"]["previous_trace_state"]["region"]),
                    "trace_after_update_region": int(swapped_update["region"]),
                    "expected_trace_after_update_region": expected,
                    "gate_pass": int(changed and aligned),
                    "note": f"swapped inspection content follows content {label}",
                }
            )
    return {
        "inspection_content_sensitivity": mean_or_zero(sensitivities),
        "inspection_swap_update_change_rate": mean_or_zero(change_rates),
        "swapped_update_alignment": mean_or_zero(alignments),
        "wrong_content_update_error": 1.0 - mean_or_zero(alignments),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
