from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np

from .features import extract_blob_centers
from .inspect_policy import select_region_from_logits
from .model_io import make_model_batch


B2_GATES = {
    "delayed_checkpoint_accuracy": 0.75,
    "multi_delay_stability": 0.70,
    "early_saliency_rejection": 0.75,
    "delay_ood_generalization": 0.70,
    "causal_trace_intervention_drop": 0.20,
    "non_trace_stability": 0.70,
    "delayed_endpoint_shift": 0.25,
}

B2_WEIGHTS = {
    "delayed_checkpoint_accuracy": 0.20,
    "multi_delay_stability": 0.15,
    "early_saliency_rejection": 0.15,
    "delay_ood_generalization": 0.20,
    "causal_trace_intervention_drop": 0.20,
    "non_trace_stability": 0.10,
}

B2_SUMMARY_KEYS = [
    "model",
    "seed",
    "delayed_checkpoint_accuracy",
    "multi_delay_stability",
    "early_saliency_rejection",
    "delay_ood_generalization",
    "causal_trace_intervention_drop",
    "non_trace_stability",
    "delayed_endpoint_shift",
    "trace_over_control_ratio",
    "b2_delayed_score",
]


def delayed_checkpoint_accuracy(pred_region: int, true_region: int) -> float:
    return 1.0 if int(pred_region) == int(true_region) else 0.0


def early_saliency_rejection(pred_region: int, saliency_region: int) -> float:
    return 1.0 if int(pred_region) != int(saliency_region) else 0.0


def multi_delay_stability(pred_region: int, best_region: int, delay_values: dict[Any, float]) -> float:
    return 1.0 if int(pred_region) == int(best_region) and bool(delay_values) else 0.0


def delay_ood_generalization(results_by_delay: dict[int, float]) -> float:
    values = [float(value) for value in results_by_delay.values()]
    return float(mean(values)) if values else 0.0


def delayed_endpoint_shift(base_future: Any, trace_removed_future: Any) -> float:
    base = np.asarray(base_future, dtype=np.float32)
    altered = np.asarray(trace_removed_future, dtype=np.float32)
    if base.size == 0 or altered.size == 0:
        return 0.0
    base_centers = extract_blob_centers(base[-1])
    altered_centers = extract_blob_centers(altered[-1])
    if len(base_centers) and len(altered_centers):
        distances = [float(np.min(np.linalg.norm(altered_centers - center, axis=1))) for center in base_centers]
        return float(np.clip(np.mean(distances) / 8.0, 0.0, 1.0))
    return float(np.clip(np.mean(np.abs(base[-1] - altered[-1])) * 4.0, 0.0, 1.0))


def evaluate_b2_behavior(model: Any, datasets: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    delayed_rows = []
    saliency_rows = []
    for idx, episode in enumerate(datasets.get("delayed", [])):
        pred = predict_region(model, episode, config)
        gt = episode["ground_truth"]
        correct = delayed_checkpoint_accuracy(pred, gt["true_delayed_checkpoint_region"])
        reject = early_saliency_rejection(pred, gt["early_saliency_region"])
        delayed_rows.append(correct)
        saliency_rows.append(reject)
        records.append(
            {
                "episode_id": idx,
                "episode_type": gt.get("episode_type", ""),
                "delay": gt.get("delay", ""),
                "true_delayed_checkpoint_region": int(gt["true_delayed_checkpoint_region"]),
                "early_saliency_region": int(gt["early_saliency_region"]),
                "predicted_region": int(pred),
                "selected_delay": "",
                "correct": int(correct),
                "ood_type": "",
                "intervention_type": "",
                "base_endpoint": "",
                "intervened_endpoint": "",
                "endpoint_shift": "",
            }
        )

    multi_rows = []
    for idx, episode in enumerate(datasets.get("multi", [])):
        pred = predict_region(model, episode, config)
        gt = episode["ground_truth"]
        score = multi_delay_stability(pred, gt["best_region"], gt["delay_values"])
        multi_rows.append(score)
        records.append(
            {
                "episode_id": idx,
                "episode_type": gt.get("episode_type", ""),
                "delay": gt.get("best_delay", ""),
                "true_delayed_checkpoint_region": int(gt["best_region"]),
                "early_saliency_region": int(gt["early_saliency_region"]),
                "predicted_region": int(pred),
                "selected_delay": selected_delay_from_region(pred, gt.get("candidate_delays", []), gt.get("candidate_regions", [])),
                "correct": int(score),
                "ood_type": "",
                "intervention_type": "",
                "base_endpoint": "",
                "intervened_endpoint": "",
                "endpoint_shift": "",
            }
        )

    by_delay: dict[int, list[float]] = {}
    for idx, episode in enumerate(datasets.get("ood", [])):
        pred = predict_region(model, episode, config)
        gt = episode["ground_truth"]
        delay = int(gt["heldout_delay"])
        correct = delayed_checkpoint_accuracy(pred, gt["true_delayed_checkpoint_region"])
        by_delay.setdefault(delay, []).append(correct)
        records.append(
            {
                "episode_id": idx,
                "episode_type": gt.get("episode_type", ""),
                "delay": delay,
                "true_delayed_checkpoint_region": int(gt["true_delayed_checkpoint_region"]),
                "early_saliency_region": int(gt["early_saliency_region"]),
                "predicted_region": int(pred),
                "selected_delay": "",
                "correct": int(correct),
                "ood_type": "heldout_delay",
                "intervention_type": "",
                "base_endpoint": "",
                "intervened_endpoint": "",
                "endpoint_shift": "",
            }
        )
    results_by_delay = {delay: _mean(values) for delay, values in by_delay.items()}

    return (
        {
            "delayed_checkpoint_accuracy": _mean(delayed_rows),
            "multi_delay_stability": _mean(multi_rows),
            "early_saliency_rejection": _mean(saliency_rows),
            "delay_ood_generalization": delay_ood_generalization(results_by_delay),
        },
        records,
    )


def b2_delayed_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    gates = {**B2_GATES, **(gates or {})}
    for key in B2_GATES:
        if float(metrics.get(key, 0.0)) < float(gates.get(key, 0.0)):
            return 0.0
    score = 0.0
    for key, weight in B2_WEIGHTS.items():
        value = float(metrics.get(key, 0.0))
        if key == "causal_trace_intervention_drop":
            value = min(value / float(gates["causal_trace_intervention_drop"]), 1.0)
        score += weight * min(max(value, 0.0), 1.0)
    return float(score)


def predict_region(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> int:
    output = model.forward(make_model_batch(episode, config))
    return int(select_region_from_logits(output.get("inspection_logits")))


def selected_delay_from_region(pred_region: int, delays: list[int], regions: list[int]) -> int | str:
    for delay, region in zip(delays, regions):
        if int(region) == int(pred_region):
            return int(delay)
    return ""


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0
