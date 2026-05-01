from __future__ import annotations

from typing import Any

import numpy as np

from .b41_action_scorers import family_trace_action_scores, selected_action
from .b4_intervention_policy import trace_family_for_model, trace_guided_intervention_policy


POLICY_SOURCE = {
    "recurrent": "recurrent_private_trace_action",
    "field": "field_private_trace_action",
    "schema": "schema_private_trace_action",
}


def collect_action_policy_provenance(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    records = []
    name = model_name or str(getattr(model, "name", ""))
    for episode_id, episode in enumerate(episodes):
        policy = trace_guided_intervention_policy(model, episode, config)
        action = policy["action"]
        trace_family = trace_family_for_model(model)
        candidate_scores = family_trace_action_scores(model, episode, config, trace_family)
        selected_type, selected_region = selected_action(candidate_scores)
        provenance = policy.get("provenance", {})
        records.append(
            {
                "record_kind": "action_provenance",
                "model": name,
                "seed": seed,
                "episode_id": episode_id,
                "audit_type": "action_policy_provenance",
                "predicted_action_type": action["action_type"],
                "predicted_region": int(action["region_id"]),
                "oracle_action_type": episode["ground_truth"]["oracle_best_action"]["action_type"],
                "oracle_region": int(episode["ground_truth"]["oracle_best_action"]["region_id"]),
                "family": episode["ground_truth"].get("b4_family", ""),
                "trace_family": trace_family,
                "policy_source": POLICY_SOURCE.get(trace_family, "unknown"),
                "shared_action_policy_used": bool(provenance.get("shared_action_policy_used", False)),
                "private_trace_action_score_used": bool(provenance.get("private_trace_action_score_used", False)),
                "fallback_used": bool(provenance.get("fallback_used", False)),
                "oracle_value_used": bool(provenance.get("oracle_intervention_value_used", False) or provenance.get("oracle_best_action_used", False)),
                "fixed_action_policy_used": False,
                "action_score": float(candidate_scores.get((selected_type, selected_region), 0.0)),
                "region_score": float(policy.get("region_score", 0.0)),
                "action_type_score": float(policy.get("action_type_score", 0.0)),
                "candidate_action_scores": len(candidate_scores),
                "gate_pass": int(not bool(provenance.get("shared_action_policy_used", False))),
                "note": "private trace action policy provenance",
            }
        )
    return records


def summarize_action_policy_provenance(provenance_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    if not provenance_records:
        return {
            "shared_action_policy_usage_rate": 1.0,
            "private_trace_action_score_usage_rate": 0.0,
            "fallback_usage_rate": 1.0,
            "oracle_value_usage_rate": 1.0,
            "fixed_action_policy_usage_rate": 1.0,
            "unknown_policy_source_rate": 1.0,
        }
    return {
        "shared_action_policy_usage_rate": mean_bool(provenance_records, "shared_action_policy_used"),
        "private_trace_action_score_usage_rate": mean_bool(provenance_records, "private_trace_action_score_used"),
        "fallback_usage_rate": mean_bool(provenance_records, "fallback_used"),
        "oracle_value_usage_rate": mean_bool(provenance_records, "oracle_value_used"),
        "fixed_action_policy_usage_rate": mean_bool(provenance_records, "fixed_action_policy_used"),
        "unknown_policy_source_rate": float(np.mean([1.0 if row.get("policy_source") == "unknown" else 0.0 for row in provenance_records])),
    }


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([1.0 if bool(row.get(key, False)) else 0.0 for row in rows])) if rows else 0.0

