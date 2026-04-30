from __future__ import annotations

from statistics import mean, pstdev
from typing import Any


B21A_GATES = {
    "max_model_score_std_for_warning": 0.005,
    "min_per_attack_variance": 0.001,
    "max_random_baseline_score": 0.20,
    "min_oracle_score": 0.95,
    "max_leakage_score": 0.00,
    "min_intervention_applicability_rate": 0.95,
    "min_trace_family_ablation_drop": 0.20,
    "min_no_trace_ablation_drop": 0.30,
    "max_predicted_region_gt_correlation_without_model": 0.20,
}

B21A_SUMMARY_KEYS = [
    "score_degeneracy_detected",
    "all_attacks_identical_flag",
    "exact_same_score_all_models_all_seeds",
    "cross_model_exact_prediction_match_rate",
    "gt_region_match_rate",
    "saliency_region_match_rate",
    "leakage_count",
    "intervention_applicability_rate",
    "fallback_rate",
    "structure_changed_rate",
    "output_changed_rate",
    "random_b21_score",
    "oracle_b21_score",
    "trace_family_ablation_drop",
    "no_trace_ablation_drop",
    "b21a_degeneracy_audit_score",
]


def b21a_degeneracy_audit_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    gates = {**B21A_GATES, **(gates or {})}
    if float(metrics.get("leakage_count", 0.0)) > float(gates["max_leakage_score"]):
        return 0.0
    if float(metrics.get("intervention_applicability_rate", 0.0)) < float(gates["min_intervention_applicability_rate"]):
        return 0.0
    if float(metrics.get("random_b21_score", 0.0)) > float(gates["max_random_baseline_score"]):
        return 0.0
    if float(metrics.get("oracle_b21_score", 0.0)) < float(gates["min_oracle_score"]):
        return 0.0
    if float(metrics.get("trace_family_ablation_drop", 0.0)) < float(gates["min_trace_family_ablation_drop"]):
        return 0.0
    if float(metrics.get("no_trace_ablation_drop", 0.0)) < float(gates["min_no_trace_ablation_drop"]):
        return 0.0
    exact_match = float(metrics.get("cross_model_exact_prediction_match_rate", 0.0))
    all_identical = float(metrics.get("all_attacks_identical_flag", 0.0)) > 0.5
    gt_match = float(metrics.get("gt_region_match_rate", 0.0))
    if exact_match >= 0.98 and all_identical and gt_match < 0.95:
        return 0.0
    return 1.0


def population_std(values: list[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = -1) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default
