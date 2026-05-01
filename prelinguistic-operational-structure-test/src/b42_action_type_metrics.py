from __future__ import annotations

from collections import Counter


B42_GATES = {
    "fixed_action_type_rate": 0.70,
    "action_type_accuracy": 0.70,
    "region_accuracy": 0.75,
    "joint_region_action_accuracy": 0.70,
    "correct_region_wrong_action_penalty": 0.25,
    "action_type_counterfactual_sensitivity": 0.25,
    "family_action_diversity": 0.50,
    "family_action_mapping_accuracy": 0.70,
    "action_type_shift_after_trace_ablation": 0.20,
    "action_type_ablation_drop": 0.20,
    "region_stability_after_action_ablation": 0.70,
    "gain_over_fixed_action_baseline": 0.15,
    "gain_over_random_action_type": 0.20,
    "gain_over_saliency": 0.15,
    "gain_over_short_horizon": 0.15,
    "action_type_ood_accuracy": 0.65,
    "oracle_action_type_score": 0.95,
    "value_leakage_count": 0,
}

B42_SUMMARY_KEYS = [
    "model",
    "seed",
    "fixed_action_type_rate",
    "action_type_accuracy",
    "region_accuracy",
    "joint_region_action_accuracy",
    "correct_region_wrong_action_penalty",
    "action_type_counterfactual_sensitivity",
    "family_action_diversity",
    "family_action_mapping_accuracy",
    "recurrent_action_accuracy",
    "field_action_accuracy",
    "schema_action_accuracy",
    "action_type_shift_after_trace_ablation",
    "action_type_ablation_drop",
    "region_stability_after_action_ablation",
    "gain_over_fixed_action_baseline",
    "gain_over_random_action_type",
    "gain_over_saliency",
    "gain_over_short_horizon",
    "action_type_ood_accuracy",
    "oracle_action_type_score",
    "value_leakage_count",
    "b42_action_type_score",
]


def fixed_action_type_rate(records: list[dict]) -> float:
    action_types = [str(row.get("predicted_action_type", "")) for row in records if row.get("predicted_action_type")]
    if not action_types:
        return 1.0
    counts = Counter(action_types)
    return max(counts.values()) / len(action_types)


def action_type_accuracy(predicted_action: dict, expected_action: dict) -> float:
    return 1.0 if str(predicted_action.get("action_type", "")) == str(expected_action.get("action_type", "")) else 0.0


def region_accuracy(predicted_action: dict, expected_action: dict) -> float:
    return 1.0 if int(predicted_action.get("region_id", -1)) == int(expected_action.get("region_id", -2)) else 0.0


def joint_region_action_accuracy(predicted_action: dict, expected_action: dict) -> float:
    return action_type_accuracy(predicted_action, expected_action) * region_accuracy(predicted_action, expected_action)


def correct_region_wrong_action_penalty(correct_value: float, wrong_action_value: float) -> float:
    return max(0.0, float(correct_value) - float(wrong_action_value))


def family_action_diversity(records: list[dict]) -> float:
    action_types = {str(row.get("predicted_action_type", "")) for row in records if row.get("predicted_action_type")}
    return len(action_types) / 4.0


def gain_over_fixed_action_baseline(model_score: float, fixed_baseline_score: float) -> float:
    return float(model_score) - float(fixed_baseline_score)


def b42_action_type_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = {**B42_GATES, **(gates or {})}
    min_checks = [
        "action_type_accuracy",
        "region_accuracy",
        "joint_region_action_accuracy",
        "correct_region_wrong_action_penalty",
        "action_type_counterfactual_sensitivity",
        "family_action_diversity",
        "family_action_mapping_accuracy",
        "action_type_shift_after_trace_ablation",
        "action_type_ablation_drop",
        "region_stability_after_action_ablation",
        "gain_over_fixed_action_baseline",
        "gain_over_random_action_type",
        "gain_over_saliency",
        "gain_over_short_horizon",
        "action_type_ood_accuracy",
        "oracle_action_type_score",
    ]
    for key in min_checks:
        if float(metrics.get(key, 0.0)) < float(gates[key]):
            return 0.0
    if float(metrics.get("fixed_action_type_rate", 1.0)) > float(gates["fixed_action_type_rate"]):
        return 0.0
    if float(metrics.get("value_leakage_count", 1.0)) > float(gates["value_leakage_count"]):
        return 0.0
    weights = {
        "action_type_accuracy": 0.18,
        "joint_region_action_accuracy": 0.16,
        "correct_region_wrong_action_penalty": 0.14,
        "action_type_counterfactual_sensitivity": 0.14,
        "family_action_mapping_accuracy": 0.14,
        "action_type_ablation_drop": 0.12,
        "gain_over_fixed_action_baseline": 0.07,
        "action_type_ood_accuracy": 0.05,
    }
    return float(sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items()))

