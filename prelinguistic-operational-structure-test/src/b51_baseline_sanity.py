from __future__ import annotations

from typing import Any

from .b5_closed_loop_baselines import evaluate_b5_baselines


def run_b51_baseline_sanity_check(episodes: list[dict[str, Any]], config: dict[str, Any], model_score: float = 1.0, seed: int = 0) -> tuple[dict[str, float], list[dict[str, Any]]]:
    baseline_metrics, records = evaluate_b5_baselines(episodes, config, seed)
    scripted_update_score = 1.0
    metrics = {
        **baseline_metrics,
        "fixed_inspect_then_intervene_score": baseline_metrics["inspect_always_score"],
        "scripted_update_then_intervene_score": scripted_update_score,
        "model_gain_over_inspect_always": model_score - baseline_metrics["inspect_always_score"],
        "model_gain_over_intervene_immediately": model_score - baseline_metrics["intervene_immediately_score"],
        "model_gain_over_saliency": model_score - baseline_metrics["saliency_closed_loop_score"],
        "model_gain_over_short_horizon": model_score - baseline_metrics["short_horizon_closed_loop_score"],
        "model_gain_over_scripted_update": model_score - scripted_update_score,
    }
    for record in records:
        record["record_kind"] = "baseline_sanity"
    return metrics, records
