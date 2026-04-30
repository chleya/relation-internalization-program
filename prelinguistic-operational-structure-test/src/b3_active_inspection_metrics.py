from __future__ import annotations

from statistics import mean


B3_GATES = {
    "trace_guided_inspection_accuracy": 0.75,
    "trace_vs_saliency_rejection": 0.75,
    "delayed_information_gain": 0.20,
    "inspection_value_gain_over_random": 0.20,
    "inspection_value_gain_over_saliency": 0.15,
    "inspection_value_gain_over_short_horizon": 0.15,
    "trace_ablation_inspection_drop": 0.20,
    "delay_ood_inspection_accuracy": 0.65,
    "oracle_inspection_score": 0.95,
    "random_inspection_score": 0.25,
}

B3_SUMMARY_KEYS = [
    "model",
    "seed",
    "trace_guided_inspection_accuracy",
    "trace_vs_saliency_rejection",
    "delayed_information_gain",
    "relative_information_gain",
    "inspection_value_gain_over_random",
    "inspection_value_gain_over_saliency",
    "inspection_value_gain_over_short_horizon",
    "base_trace_guided_inspection_accuracy",
    "trace_ablated_inspection_accuracy",
    "trace_ablation_inspection_drop",
    "delay_ood_inspection_accuracy",
    "oracle_inspection_score",
    "random_inspection_score",
    "saliency_inspection_score",
    "short_horizon_inspection_score",
    "b3_active_inspection_score",
]


def trace_guided_inspection_accuracy(predicted_region: int, oracle_best_region: int, tolerance: int = 0) -> float:
    return 1.0 if int(predicted_region) == int(oracle_best_region) else 0.0


def trace_vs_saliency_rejection(predicted_region: int, saliency_region: int, true_trace_region: int) -> float:
    return 1.0 if int(predicted_region) != int(saliency_region) and int(predicted_region) == int(true_trace_region) else 0.0


def inspection_value_gain_over_baseline(model_gain: float, baseline_gain: float) -> float:
    return float(model_gain) - float(baseline_gain)


def trace_ablation_inspection_drop(base_accuracy: float, trace_ablated_accuracy: float) -> float:
    return max(0.0, float(base_accuracy) - float(trace_ablated_accuracy))


def delay_ood_inspection_accuracy(results_by_delay: dict[int, float]) -> float:
    return float(mean(results_by_delay.values())) if results_by_delay else 0.0


def b3_active_inspection_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = {**B3_GATES, **(gates or {})}
    min_gates = [
        "trace_guided_inspection_accuracy",
        "trace_vs_saliency_rejection",
        "delayed_information_gain",
        "inspection_value_gain_over_random",
        "inspection_value_gain_over_saliency",
        "inspection_value_gain_over_short_horizon",
        "trace_ablation_inspection_drop",
        "delay_ood_inspection_accuracy",
        "oracle_inspection_score",
    ]
    for key in min_gates:
        if float(metrics.get(key, 0.0)) < float(gates[key]):
            return 0.0
    if float(metrics.get("random_inspection_score", 1.0)) > float(gates["random_inspection_score"]):
        return 0.0
    weights = {
        "trace_guided_inspection_accuracy": 0.20,
        "trace_vs_saliency_rejection": 0.15,
        "delayed_information_gain": 0.20,
        "inspection_value_gain_over_saliency": 0.15,
        "inspection_value_gain_over_short_horizon": 0.15,
        "trace_ablation_inspection_drop": 0.15,
    }
    return float(sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items()))
