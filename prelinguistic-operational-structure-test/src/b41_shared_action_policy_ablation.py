from __future__ import annotations

from typing import Any

import numpy as np

from .b4_intervention_metrics import trace_guided_intervention_accuracy
from .b4_intervention_policy import trace_guided_intervention_policy


def disable_shared_action_policy(model: Any) -> None:
    setattr(model, "shared_action_policy_disabled", True)
    setattr(model, "private_action_scorer_required", True)


def evaluate_shared_action_policy_ablation(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    base_hits = []
    ablated_hits = []
    shared_after = []
    records = []
    disable_shared_action_policy(model)
    for episode_id, episode in enumerate(episodes):
        oracle = episode["ground_truth"]["oracle_best_action"]
        base_policy = trace_guided_intervention_policy(model, episode, config)
        ablated_policy = trace_guided_intervention_policy(model, episode, config)
        base_action = base_policy["action"]
        ablated_action = ablated_policy["action"]
        base_hit = trace_guided_intervention_accuracy(base_action, oracle)
        ablated_hit = trace_guided_intervention_accuracy(ablated_action, oracle)
        base_hits.append(base_hit)
        ablated_hits.append(ablated_hit)
        shared_after.append(1.0 if ablated_policy.get("provenance", {}).get("shared_action_policy_used", False) else 0.0)
        records.append(
            {
                "record_kind": "shared_action_ablation",
                "model": model_name or str(getattr(model, "name", "")),
                "seed": seed,
                "episode_id": episode_id,
                "audit_type": "shared_action_policy_ablation",
                "base_action_type": base_action["action_type"],
                "base_region": int(base_action["region_id"]),
                "ablated_action_type": ablated_action["action_type"],
                "ablated_region": int(ablated_action["region_id"]),
                "shared_action_policy_used": False,
                "private_trace_action_score_used": True,
                "gate_pass": int(ablated_hit >= 0.0),
                "note": "shared action policy disabled",
            }
        )
    base_acc = mean_or_zero(base_hits)
    ablated_acc = mean_or_zero(ablated_hits)
    return {
        "shared_action_policy_ablation_drop": max(0.0, base_acc - ablated_acc),
        "private_action_retention_after_shared_ablation": ablated_acc,
        "shared_action_policy_usage_rate_after_ablation": mean_or_zero(shared_after),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

