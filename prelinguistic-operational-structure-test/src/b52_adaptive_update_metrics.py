from __future__ import annotations

from typing import Any

import numpy as np


B52_SUMMARY_KEYS = [
    "model",
    "seed",
    "value_leakage_count",
    "inspection_content_sensitivity",
    "inspection_swap_update_change_rate",
    "counterfactual_update_switch_rate",
    "counterfactual_plan_switch_rate",
    "same_initial_different_info_plan_divergence",
    "post_update_plan_divergence",
    "post_update_intervention_change_rate",
    "model_update_score",
    "scripted_update_score",
    "model_gain_over_scripted_update",
    "update_specificity_over_scripted",
    "feedback_content_sensitivity",
    "contradictory_feedback_revision_accuracy",
    "delayed_feedback_revision_accuracy",
    "model_feedback_score",
    "scripted_feedback_score",
    "model_gain_over_scripted_feedback",
    "feedback_specificity_over_scripted",
    "revision_specific_ablation_drop",
    "update_path_ablation_drop",
    "feedback_path_ablation_drop",
    "non_revision_path_stability",
    "cross_model_exact_plan_match_rate",
    "exact_all_model_same_plan_rate",
    "oracle_adaptive_update_score",
    "random_update_score",
    "b52_adaptive_update_score",
]

B52_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "pair_id",
    "test_type",
    "initial_observation_signature",
    "inspection_content_signature",
    "counterfactual_type",
    "trace_before_region",
    "trace_after_update_region",
    "expected_trace_after_update_region",
    "intervention_before_update",
    "intervention_after_update",
    "plan_signature_before",
    "plan_signature_after",
    "consequence_type",
    "feedback_revision_region",
    "expected_feedback_revision_region",
    "scripted_update_region",
    "scripted_feedback_region",
    "ablation_type",
    "gate_pass",
    "note",
]


def inspection_content_sensitivity(records: list[dict[str, Any]]) -> float:
    return mean_or_zero([float(row.get("gate_pass", 0)) for row in records if row.get("test_type") == "inspection_content_swap"])


def counterfactual_update_switch_rate(records: list[dict[str, Any]]) -> float:
    return mean_or_zero([float(row.get("gate_pass", 0)) for row in records if row.get("test_type") == "counterfactual_inspection"])


def same_initial_different_info_plan_divergence(records: list[dict[str, Any]]) -> float:
    return mean_or_zero([float(row.get("gate_pass", 0)) for row in records if row.get("test_type") == "plan_divergence"])


def feedback_content_sensitivity(records: list[dict[str, Any]]) -> float:
    return mean_or_zero([float(row.get("gate_pass", 0)) for row in records if row.get("test_type") == "feedback_stress"])


def model_gain_over_scripted_update(model_score: float, scripted_score: float) -> float:
    return float(model_score) - float(scripted_score)


def model_gain_over_scripted_feedback(model_score: float, scripted_score: float) -> float:
    return float(model_score) - float(scripted_score)


def b52_adaptive_update_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = gates or {}
    max_checks = {
        "value_leakage_count": gates.get("value_leakage_count", 0.0),
        "oracle_plan_usage_rate": gates.get("oracle_plan_usage_rate", 0.0),
        "oracle_trace_update_usage_rate": gates.get("oracle_trace_update_usage_rate", 0.0),
        "oracle_feedback_revision_usage_rate": gates.get("oracle_feedback_revision_usage_rate", 0.0),
        "cross_model_exact_plan_match_rate": gates.get("cross_model_exact_plan_match_rate_max", 0.80),
        "exact_all_model_same_plan_rate": gates.get("exact_all_model_same_plan_rate_max", 0.80),
        "random_update_score": gates.get("random_update_score_max", 0.25),
    }
    for key, threshold in max_checks.items():
        if float(metrics.get(key, 0.0)) > float(threshold):
            return 0.0
    min_checks = {
        "inspection_content_sensitivity": gates.get("inspection_content_sensitivity", 0.70),
        "inspection_swap_update_change_rate": gates.get("inspection_swap_update_change_rate", 0.50),
        "counterfactual_update_switch_rate": gates.get("counterfactual_update_switch_rate", 0.50),
        "same_initial_different_info_plan_divergence": gates.get("same_initial_different_info_plan_divergence", 0.50),
        "post_update_plan_divergence": gates.get("post_update_plan_divergence", 0.50),
        "post_update_intervention_change_rate": gates.get("post_update_intervention_change_rate", 0.50),
        "model_gain_over_scripted_update": gates.get("model_gain_over_scripted_update", 0.15),
        "update_specificity_over_scripted": gates.get("update_specificity_over_scripted", 1.50),
        "update_specificity_over_shuffled": gates.get("update_specificity_over_shuffled", 1.50),
        "feedback_content_sensitivity": gates.get("feedback_content_sensitivity", 0.65),
        "contradictory_feedback_revision_accuracy": gates.get("contradictory_feedback_revision_accuracy", 0.65),
        "delayed_feedback_revision_accuracy": gates.get("delayed_feedback_revision_accuracy", 0.60),
        "model_gain_over_scripted_feedback": gates.get("model_gain_over_scripted_feedback", 0.15),
        "feedback_specificity_over_scripted": gates.get("feedback_specificity_over_scripted", 1.50),
        "revision_specific_ablation_drop": gates.get("revision_specific_ablation_drop", 0.20),
        "update_path_ablation_drop": gates.get("update_path_ablation_drop", 0.20),
        "feedback_path_ablation_drop": gates.get("feedback_path_ablation_drop", 0.20),
        "non_revision_path_stability": gates.get("non_revision_path_stability", 0.70),
        "oracle_adaptive_update_score": gates.get("oracle_adaptive_update_score", 0.95),
    }
    for key, threshold in min_checks.items():
        if float(metrics.get(key, 0.0)) < float(threshold):
            return 0.0
    weights = {
        "inspection_content_sensitivity": 0.15,
        "counterfactual_update_switch_rate": 0.15,
        "same_initial_different_info_plan_divergence": 0.15,
        "model_gain_over_scripted_update": 0.15,
        "feedback_content_sensitivity": 0.15,
        "model_gain_over_scripted_feedback": 0.15,
        "revision_specific_ablation_drop": 0.10,
    }
    return float(sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items()))


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
