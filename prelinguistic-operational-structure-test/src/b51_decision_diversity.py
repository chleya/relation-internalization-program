from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def compute_decision_diversity(b5_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    records = policy_records(b5_records)
    decisions = []
    for record in records:
        inspect = "inspect" if int(record.get("inspect_skipped", 0)) == 0 else "skip_inspect"
        action_type = str(record.get("predicted_intervention_action_type", "do_nothing"))
        intervene = "skip_intervention" if action_type == "do_nothing" else "intervene"
        decisions.append((inspect, intervene))
    possible = 4
    return {"decision_diversity_score": len(set(decisions)) / possible if decisions else 0.0}


def compute_shortcut_rates(b5_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    records = policy_records(b5_records)
    inspect_always = []
    intervene_immediately = []
    inspect_then_intervene = []
    do_nothing = []
    skip_inspect_not_needed = []
    skip_intervention_not_needed = []
    for record in records:
        inspect_skipped = int(record.get("inspect_skipped", 0)) == 1
        action_type = str(record.get("predicted_intervention_action_type", "do_nothing"))
        intervened = action_type != "do_nothing"
        needs_inspection = int(record.get("needs_inspection", 0)) == 1
        no_action_episode = str(record.get("episode_type", "")) == "no_action"
        inspect_always.append(0.0 if inspect_skipped else 1.0)
        intervene_immediately.append(1.0 if inspect_skipped and intervened else 0.0)
        inspect_then_intervene.append(1.0 if (not inspect_skipped and intervened) else 0.0)
        do_nothing.append(1.0 if inspect_skipped and not intervened else 0.0)
        if not needs_inspection:
            skip_inspect_not_needed.append(1.0 if inspect_skipped else 0.0)
        if no_action_episode:
            skip_intervention_not_needed.append(1.0 if not intervened else 0.0)
    return {
        "inspect_always_rate": mean_or_zero(inspect_always),
        "intervene_immediately_rate": mean_or_zero(intervene_immediately),
        "always_inspect_then_intervene_rate": mean_or_zero(inspect_then_intervene),
        "do_nothing_rate": mean_or_zero(do_nothing),
        "skip_inspect_when_not_needed_rate": mean_or_zero(skip_inspect_not_needed),
        "skip_intervention_when_not_needed_rate": mean_or_zero(skip_intervention_not_needed),
    }


def evaluate_episode_type_conditioned_decisions(b5_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    records = policy_records(b5_records)
    hits = []
    for record in records:
        episode_type = str(record.get("episode_type", ""))
        inspect_skipped = int(record.get("inspect_skipped", 0)) == 1
        intervened = str(record.get("predicted_intervention_action_type", "do_nothing")) != "do_nothing"
        expected = {
            "inspect_needed": (False, True),
            "misleading_initial_trace": (False, True),
            "intervention_clear": (True, True),
            "no_action": (True, False),
        }.get(episode_type, (True, True))
        hits.append(1.0 if (inspect_skipped, intervened) == expected else 0.0)
    return {"episode_type_conditioned_decision_accuracy": mean_or_zero(hits)}


def policy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("record_kind") == "closed_loop_policy"]


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
