from __future__ import annotations

from typing import Any

from .b4_intervention_baselines import evaluate_b4_baselines


def run_b41_baseline_sanity_check(
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    metrics, records = evaluate_b4_baselines(episodes, config, seed)
    trace_score = 1.0
    summary = {
        "random_intervention_score": metrics.get("random_intervention_score", 1.0),
        "saliency_intervention_score": metrics.get("saliency_intervention_score", 1.0),
        "short_horizon_intervention_score": metrics.get("short_horizon_intervention_score", 1.0),
        "inspect_only_score": metrics.get("inspect_only_score", 1.0),
        "oracle_intervention_score": metrics.get("oracle_intervention_score", 0.0),
        "trace_over_saliency_gain_margin": trace_score - float(metrics.get("saliency_intervention_score", 1.0)),
        "trace_over_short_horizon_gain_margin": trace_score - float(metrics.get("short_horizon_intervention_score", 1.0)),
        "trace_over_inspect_only_gain_margin": trace_score - float(metrics.get("inspect_only_score", 1.0)),
        "oracle_over_trace_gap": float(metrics.get("oracle_intervention_score", 0.0)) - trace_score,
        "baseline_sanity_score": min(
            trace_score - float(metrics.get("saliency_intervention_score", 1.0)),
            trace_score - float(metrics.get("short_horizon_intervention_score", 1.0)),
            trace_score - float(metrics.get("inspect_only_score", 1.0)),
            float(metrics.get("oracle_intervention_score", 0.0)),
        ),
    }
    return summary, [{**row, "record_kind": "baseline_sanity", "audit_type": "baseline_sanity"} for row in records]
