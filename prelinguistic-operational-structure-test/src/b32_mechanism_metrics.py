from __future__ import annotations

from statistics import mean


B32_GATES = {
    "family_specific_inspection_accuracy": 0.70,
    "recurrent_goal_accuracy": 0.70,
    "field_goal_accuracy": 0.70,
    "schema_goal_accuracy": 0.70,
    "mechanism_disagreement_rate": 0.50,
    "cross_model_same_region_rate": 0.70,
    "task_conditioned_switch_accuracy": 0.70,
    "inspect_value_decomposition_alignment": 0.70,
    "family_specific_trace_ablation_drop": 0.20,
    "non_target_family_stability": 0.70,
    "gain_over_random": 0.20,
    "gain_over_saliency": 0.15,
    "gain_over_short_horizon": 0.15,
    "oracle_family_inspection_score": 0.95,
}

B32_SUMMARY_KEYS = [
    "model",
    "seed",
    "family_specific_inspection_accuracy",
    "recurrent_goal_accuracy",
    "field_goal_accuracy",
    "schema_goal_accuracy",
    "mechanism_disagreement_rate",
    "cross_model_same_region_rate",
    "task_conditioned_switch_accuracy",
    "inspect_value_decomposition_alignment",
    "recurrent_value_alignment",
    "field_value_alignment",
    "schema_value_alignment",
    "family_specific_trace_ablation_drop",
    "recurrent_trace_ablation_drop",
    "field_trace_ablation_drop",
    "schema_trace_ablation_drop",
    "non_target_family_stability",
    "gain_over_random",
    "gain_over_saliency",
    "gain_over_short_horizon",
    "oracle_family_inspection_score",
    "b32_mechanism_inspection_score",
]


def family_specific_inspection_accuracy(predicted_region: int, expected_family_region: int, tolerance: int = 0) -> float:
    return 1.0 if int(predicted_region) == int(expected_family_region) else 0.0


def task_conditioned_switch_accuracy(predictions_by_goal: dict[str, int], expected_by_goal: dict[str, int]) -> float:
    if not expected_by_goal:
        return 0.0
    return float(mean(1.0 if int(predictions_by_goal.get(goal, -1)) == int(expected) else 0.0 for goal, expected in expected_by_goal.items()))


def mechanism_disagreement_rate(predictions_by_model: dict[str, int]) -> float:
    return 1.0 if len(set(int(value) for value in predictions_by_model.values())) > 1 else 0.0


def inspect_value_decomposition_alignment(selected_region: int, value_maps: dict[str, dict[int, float]], goal_family: str) -> float:
    key = {"recurrent_goal": "recurrent_value", "field_goal": "field_value", "schema_goal": "schema_value"}.get(goal_family, "combined_value")
    value_map = value_maps.get(key, {})
    if not value_map:
        return 0.0
    best = max(value_map.items(), key=lambda item: (float(item[1]), -int(item[0])))[0]
    return 1.0 if int(selected_region) == int(best) else 0.0


def cross_model_same_region_rate(predictions_by_model_and_episode: dict[int, dict[str, int]]) -> float:
    if not predictions_by_model_and_episode:
        return 1.0
    return float(mean(1.0 if len(set(row.values())) <= 1 else 0.0 for row in predictions_by_model_and_episode.values()))


def b32_mechanism_inspection_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = {**B32_GATES, **(gates or {})}
    min_gates = [
        "family_specific_inspection_accuracy",
        "recurrent_goal_accuracy",
        "field_goal_accuracy",
        "schema_goal_accuracy",
        "mechanism_disagreement_rate",
        "task_conditioned_switch_accuracy",
        "inspect_value_decomposition_alignment",
        "family_specific_trace_ablation_drop",
        "non_target_family_stability",
        "gain_over_random",
        "gain_over_saliency",
        "gain_over_short_horizon",
        "oracle_family_inspection_score",
    ]
    for key in min_gates:
        if float(metrics.get(key, 0.0)) < float(gates[key]):
            return 0.0
    if float(metrics.get("cross_model_same_region_rate", 1.0)) > float(gates["cross_model_same_region_rate"]):
        return 0.0
    weights = {
        "family_specific_inspection_accuracy": 0.20,
        "mechanism_disagreement_rate": 0.20,
        "task_conditioned_switch_accuracy": 0.20,
        "inspect_value_decomposition_alignment": 0.15,
        "family_specific_trace_ablation_drop": 0.15,
        "gain_over_saliency": 0.10,
    }
    return float(sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items()))
