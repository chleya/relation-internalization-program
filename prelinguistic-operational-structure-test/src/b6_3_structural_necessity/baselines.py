from __future__ import annotations

from typing import Any, Callable

from src.b6_2_hardening.baselines import get_baseline
from src.b6_2_hardening.policy import b62_policy

from .ablations import apply_ablation
from .structural_policy import b63_policy


BASELINE_NAMES = [
    "b62_policy",
    "b63_policy",
    "state_only",
    "mask_only",
    "trace_only",
    "random",
    "always_abstain",
    "conservative_uncertainty",
    "oracle",
]

ABLATION_POLICY_NAMES = [
    "b63_no_trace",
    "b63_shuffled_trace",
    "b63_corrupt_trace",
    "b63_no_feedback_update",
    "b63_no_history",
    "b63_no_risk_cue",
    "b63_no_candidate_search",
    "b63_no_credit_buffer",
    "b63_no_inspection_recovery",
]


def get_policy(name: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    if name == "b62_policy":
        return b62_policy
    if name == "b63_policy":
        return b63_policy
    if name in {
        "state_only",
        "mask_only",
        "trace_only",
        "random",
        "always_abstain",
        "conservative_uncertainty",
        "oracle",
    }:
        return get_baseline(name)
    ablation_map = {
        "b63_no_trace": "remove_trace",
        "b63_shuffled_trace": "shuffle_trace",
        "b63_corrupt_trace": "corrupt_trace",
        "b63_no_feedback_update": "freeze_feedback_update",
        "b63_no_history": "remove_history",
        "b63_no_risk_cue": "remove_risk_cue",
        "b63_no_candidate_search": "disable_candidate_search",
        "b63_no_credit_buffer": "disable_delayed_credit_buffer",
        "b63_no_inspection_recovery": "disable_inspection_recovery",
    }
    if name in ablation_map:
        ablation = ablation_map[name]

        def run(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            return b63_policy(apply_ablation(episode, ablation, config), config)

        return run
    raise KeyError(name)
