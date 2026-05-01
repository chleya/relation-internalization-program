from __future__ import annotations

from typing import Any

import numpy as np

from .b3_inspection_policy import trace_guided_inspection_policy


POLICY_SOURCE_BY_TRACE_FAMILY = {
    "recurrent_memory": "recurrent_private_trace_inspection",
    "field_trace": "field_private_trace_inspection",
    "schema_memory": "schema_private_trace_inspection",
}


def collect_inspection_provenance(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    model_name = str(getattr(model, "name", ""))
    for episode_id, episode in enumerate(episodes):
        policy = trace_guided_inspection_policy(model, episode, config)
        provenance = dict(policy.get("provenance", {}))
        trace_family = str(policy.get("trace_family", "unknown"))
        policy_source = POLICY_SOURCE_BY_TRACE_FAMILY.get(trace_family, str(policy.get("policy_source", "unknown")))
        scores = policy.get("candidate_region_scores", {})
        gt = episode["ground_truth"]
        records.append(
            {
                "model": model_name,
                "episode_id": episode_id,
                "delay": int(gt.get("delay", 0)),
                "inspect_region": int(policy["inspect_region"]),
                "policy_source": policy_source,
                "trace_family": trace_family,
                "inspection_score": float(policy.get("inspection_score", policy.get("trace_score", 0.0))),
                "shared_inspection_policy_used": bool(provenance.get("shared_inspection_policy_used", False)),
                "private_trace_inspection_score_used": bool(provenance.get("private_trace_inspection_score_used", False)),
                "fallback_used": bool(provenance.get("fallback_used", False)),
                "candidate_region_scores": ";".join(f"{int(k)}:{float(v):.6f}" for k, v in sorted(scores.items())),
                "true_trace_region": int(gt["true_trace_region"]),
                "saliency_region": int(gt["saliency_region"]),
                "oracle_best_inspect_region": int(gt["oracle_best_inspect_region"]),
            }
        )
    return records


def summarize_inspection_provenance(provenance_records: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, float]:
    if not provenance_records:
        return {
            "shared_inspection_policy_usage_rate": 1.0,
            "private_trace_inspection_score_usage_rate": 0.0,
            "fallback_usage_rate": 1.0,
            "unknown_policy_source_rate": 1.0,
        }
    return {
        "shared_inspection_policy_usage_rate": mean_bool(provenance_records, "shared_inspection_policy_used"),
        "private_trace_inspection_score_usage_rate": mean_bool(provenance_records, "private_trace_inspection_score_used"),
        "fallback_usage_rate": mean_bool(provenance_records, "fallback_used"),
        "unknown_policy_source_rate": float(
            np.mean([1.0 if str(row.get("policy_source", "unknown")) == "unknown" else 0.0 for row in provenance_records])
        ),
    }


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([1.0 if bool(row.get(key, False)) else 0.0 for row in rows])) if rows else 0.0
