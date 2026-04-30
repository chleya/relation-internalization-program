from __future__ import annotations

from statistics import mean


B21_GATES = {
    "false_trace_rejection": 0.75,
    "trace_swap_sensitivity": 0.70,
    "trace_deletion_specificity_ratio": 1.50,
    "multi_source_conflict_resolution": 0.70,
    "noisy_trace_robustness": 0.70,
    "trace_length_extrapolation": 0.65,
    "trace_compression_survival": 0.65,
    "true_trace_intervention_drop": 0.20,
    "non_trace_stability": 0.70,
}

B21_WEIGHTS = {
    "false_trace_rejection": 0.15,
    "trace_swap_sensitivity": 0.15,
    "trace_deletion_specificity_ratio": 0.20,
    "multi_source_conflict_resolution": 0.15,
    "noisy_trace_robustness": 0.10,
    "trace_length_extrapolation": 0.15,
    "trace_compression_survival": 0.10,
}

B21_SUMMARY_KEYS = [
    "model",
    "seed",
    "false_trace_rejection",
    "false_trace_selected_rate",
    "true_trace_selected_rate",
    "trace_swap_sensitivity",
    "trace_swap_endpoint_shift",
    "trace_deletion_specificity_ratio",
    "true_trace_intervention_drop",
    "matched_non_trace_drop",
    "non_trace_stability",
    "multi_source_conflict_resolution",
    "wrong_trace_follow_rate",
    "noisy_trace_robustness",
    "accuracy_under_mild_noise",
    "accuracy_under_medium_noise",
    "accuracy_under_strong_noise",
    "noise_degradation_slope",
    "trace_length_extrapolation",
    "delay_8_accuracy",
    "delay_10_accuracy",
    "long_delay_degradation",
    "trace_compression_survival",
    "accuracy_at_075",
    "accuracy_at_050",
    "accuracy_at_025",
    "causal_trace_retention_under_compression",
    "saliency_retention_bias",
    "b21_trace_hardening_score",
]

RATIO_KEYS = {"trace_deletion_specificity_ratio"}


def probability(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def false_trace_rejection(pred_region: int, true_region: int, false_region: int) -> float:
    return 1.0 if int(pred_region) == int(true_region) and int(pred_region) != int(false_region) else 0.0


def trace_swap_sensitivity(base_region: int, swapped_region: int, expected_region: int) -> float:
    return 1.0 if int(base_region) != int(swapped_region) and int(swapped_region) == int(expected_region) else 0.0


def trace_deletion_specificity_ratio(true_drop: float, non_trace_drop: float) -> float:
    return float(true_drop) / (float(non_trace_drop) + 1e-6)


def multi_source_conflict_resolution(pred_region: int, true_region: int, wrong_region: int) -> float:
    return 1.0 if int(pred_region) == int(true_region) and int(pred_region) != int(wrong_region) else 0.0


def noisy_trace_robustness(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def trace_length_extrapolation(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def trace_compression_survival(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def b21_trace_hardening_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    gates = {**B21_GATES, **(gates or {})}
    for key in B21_GATES:
        if float(metrics.get(key, 0.0)) < float(gates.get(key, 0.0)):
            return 0.0
    score = 0.0
    for key, weight in B21_WEIGHTS.items():
        value = float(metrics.get(key, 0.0))
        if key in RATIO_KEYS:
            value = min(value / float(gates[key]), 1.0)
        score += weight * probability(value)
    return float(score)


def mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0
