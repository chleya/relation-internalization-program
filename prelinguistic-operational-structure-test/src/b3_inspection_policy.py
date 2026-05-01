from __future__ import annotations

from typing import Any

import numpy as np

from .b23_private_selectors import select_region_with_private_selector
from .model_io import make_model_batch


def trace_guided_inspection_policy(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    batch = make_model_batch(episode, config)
    selection = select_region_with_private_selector(model, batch)
    region = int(selection["selected_region"])
    scores = selection.get("candidate_scores", {})
    shared_used = bool(selection.get("shared_selector_used", False))
    fallback_used = bool(selection.get("fallback_used", False))
    private_used = bool(selection.get("model_private_score_used", True))
    return {
        "inspect_region": region,
        "policy_source": str(selection.get("source_module", "private_trace_selector")),
        "trace_family": str(selection.get("source_trace_family", "")),
        "trace_score": float(scores.get(region, selection.get("score", 0.0))) if isinstance(scores, dict) else float(selection.get("score", 0.0)),
        "saliency_score": None,
        "short_horizon_score": None,
        "inspection_score": float(scores.get(region, selection.get("score", 0.0))) if isinstance(scores, dict) else float(selection.get("score", 0.0)),
        "candidate_region_scores": {int(key): float(value) for key, value in scores.items()} if isinstance(scores, dict) else {},
        "provenance": {
            "shared_selector_used": shared_used,
            "shared_inspection_policy_used": shared_used,
            "fallback_used": fallback_used,
            "model_private_score_used": private_used,
            "private_trace_inspection_score_used": private_used,
        },
    }


def evaluate_trace_guided_policy(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    rejection = []
    records = []
    for idx, episode in enumerate(episodes):
        policy = trace_guided_inspection_policy(model, episode, config)
        pred = int(policy["inspect_region"])
        gt = episode["ground_truth"]
        oracle = int(gt["oracle_best_inspect_region"])
        saliency = int(gt["saliency_region"])
        true_trace = int(gt["true_trace_region"])
        correct = 1.0 if pred == oracle else 0.0
        reject = 1.0 if pred != saliency and pred == true_trace else 0.0
        rows.append(correct)
        rejection.append(reject)
        records.append(
            {
                "episode_id": idx,
                "episode_type": gt.get("episode_type", ""),
                "delay": gt.get("delay", ""),
                "true_trace_region": true_trace,
                "saliency_region": saliency,
                "oracle_best_inspect_region": oracle,
                "predicted_inspect_region": pred,
                "policy_source": policy["policy_source"],
                "trace_family": policy["trace_family"],
                "gate_pass": int(correct),
            }
        )
    return {
        "trace_guided_inspection_accuracy": mean_or_zero(rows),
        "trace_vs_saliency_rejection": mean_or_zero(rejection),
    }, records


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
