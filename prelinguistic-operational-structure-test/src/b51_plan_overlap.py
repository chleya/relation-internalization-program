from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import numpy as np


def closed_loop_plan_signature(record: dict[str, Any]) -> tuple:
    inspect_region = int(record.get("predicted_inspect_region", -1))
    inspect_decision = "skip" if int(record.get("inspect_skipped", 0)) else "inspect"
    action_type = str(record.get("predicted_intervention_action_type", "do_nothing"))
    intervention_region = int(record.get("predicted_intervention_region", -1))
    intervention_decision = "skip" if action_type == "do_nothing" else "intervene"
    return (
        inspect_decision,
        None if inspect_decision == "skip" else inspect_region,
        int(record.get("trace_after_inspection_region", -1)),
        intervention_decision,
        None if intervention_decision == "skip" else action_type,
        None if intervention_decision == "skip" else intervention_region,
        int(record.get("trace_after_feedback_region", -1)),
    )


def compute_per_episode_plan_overlap(b5_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    by_episode: dict[int, list[tuple]] = defaultdict(list)
    for record in policy_records(b5_records):
        by_episode[int(record["episode_id"])].append(closed_loop_plan_signature(record))
    exact_matches = []
    pairwise_matches = []
    same_inspect_diff_intervention = []
    same_intervention_diff_update = []
    for signatures in by_episode.values():
        if len(signatures) < 2:
            continue
        all_same = len(set(signatures)) == 1
        exact_matches.append(1.0 if all_same else 0.0)
        pairs = list(combinations(signatures, 2))
        pairwise_matches.extend(1.0 if left == right else 0.0 for left, right in pairs)
        same_inspect_diff_intervention.extend(1.0 if left[:2] == right[:2] and left[3:6] != right[3:6] else 0.0 for left, right in pairs)
        same_intervention_diff_update.extend(1.0 if left[3:6] == right[3:6] and left[-1] != right[-1] else 0.0 for left, right in pairs)
    return {
        "cross_model_exact_plan_match_rate": mean_or_zero(pairwise_matches),
        "exact_all_model_same_plan_rate": mean_or_zero(exact_matches),
        "pairwise_plan_match_rate": mean_or_zero(pairwise_matches),
        "same_inspect_different_intervention_rate": mean_or_zero(same_inspect_diff_intervention),
        "same_intervention_different_update_rate": mean_or_zero(same_intervention_diff_update),
    }


def compute_fixed_closed_loop_plan_rate(b5_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    by_model: dict[str, list[tuple]] = defaultdict(list)
    for record in policy_records(b5_records):
        by_model[str(record.get("model", ""))].append(closed_loop_plan_signature(record))
    rates = []
    for signatures in by_model.values():
        counts = Counter(signatures)
        rates.append(max(counts.values()) / len(signatures) if signatures else 1.0)
    return {"fixed_closed_loop_plan_rate": mean_or_zero(rates)}


def policy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("record_kind") == "closed_loop_policy"]


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
