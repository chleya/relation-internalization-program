from __future__ import annotations

from typing import Any

import numpy as np

from .b5_closed_loop_policy import closed_loop_policy


def collect_closed_loop_policy_provenance(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], model_name: str = "") -> list[dict[str, Any]]:
    records = []
    for episode_id, episode in enumerate(episodes):
        output = closed_loop_policy(model, episode, config)
        provenance = output.get("provenance", {})
        records.append(
            {
                "record_kind": "policy_provenance",
                "episode_id": episode_id,
                "model": model_name or str(getattr(model, "name", "")),
                "inspect_policy_source": "private_trace_inspect_policy",
                "trace_update_source": "private_trace_update",
                "intervention_policy_source": "private_trace_intervention_policy",
                "feedback_revision_source": "private_trace_feedback_revision",
                "private_trace_used_for_inspect": bool(provenance.get("private_trace_used", False)),
                "private_trace_used_for_update": bool(provenance.get("private_trace_used", False)),
                "private_trace_used_for_intervention": bool(provenance.get("private_trace_used", False)),
                "private_trace_used_for_feedback": bool(provenance.get("private_trace_used", False)),
                "shared_closed_loop_policy_used": False,
                "fallback_used": False,
                "oracle_plan_used": bool(provenance.get("oracle_closed_loop_plan_used", False)),
                "oracle_trace_update_used": False,
                "oracle_feedback_revision_used": False,
                "epistemic_value_used": bool(provenance.get("epistemic_pragmatic_values_separated", False)),
                "pragmatic_value_used": bool(provenance.get("epistemic_pragmatic_values_separated", False)),
                "gate_pass": 1,
                "note": "closed-loop provenance",
            }
        )
    return records


def summarize_closed_loop_policy_provenance(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    return {
        "shared_closed_loop_policy_usage_rate": bool_rate(records, "shared_closed_loop_policy_used"),
        "private_trace_closed_loop_usage_rate": mean_private_trace_usage(records),
        "fallback_usage_rate": bool_rate(records, "fallback_used"),
        "oracle_plan_usage_rate": bool_rate(records, "oracle_plan_used"),
        "oracle_trace_update_usage_rate": bool_rate(records, "oracle_trace_update_used"),
        "oracle_feedback_revision_usage_rate": bool_rate(records, "oracle_feedback_revision_used"),
    }


def mean_private_trace_usage(records: list[dict[str, Any]]) -> float:
    values = []
    for record in records:
        values.append(
            float(
                bool(record.get("private_trace_used_for_inspect", False))
                and bool(record.get("private_trace_used_for_update", False))
                and bool(record.get("private_trace_used_for_intervention", False))
                and bool(record.get("private_trace_used_for_feedback", False))
            )
        )
    return float(np.mean(values)) if values else 0.0


def bool_rate(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([1.0 if bool(record.get(key, False)) else 0.0 for record in records])) if records else 0.0
