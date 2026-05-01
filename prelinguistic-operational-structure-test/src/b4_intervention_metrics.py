from __future__ import annotations


B4_GATES = {
    "trace_guided_intervention_accuracy": 0.75,
    "intervention_region_accuracy": 0.75,
    "action_type_accuracy": 0.70,
    "outcome_improvement": 0.20,
    "intervention_vs_inspection_gain": 0.15,
    "wrong_region_penalty_sensitivity": 0.20,
    "family_specific_intervention_accuracy": 0.70,
    "recurrent_intervention_accuracy": 0.70,
    "field_intervention_accuracy": 0.70,
    "schema_intervention_accuracy": 0.70,
    "trace_ablation_intervention_drop": 0.20,
    "delay_ood_intervention_accuracy": 0.65,
    "gain_over_random": 0.20,
    "gain_over_saliency": 0.15,
    "gain_over_short_horizon": 0.15,
    "gain_over_inspect_only": 0.15,
    "oracle_intervention_score": 0.95,
    "random_intervention_score_max": 0.25,
}

B4_SUMMARY_KEYS = [
    "model",
    "seed",
    "trace_guided_intervention_accuracy",
    "intervention_region_accuracy",
    "action_type_accuracy",
    "outcome_improvement",
    "intervention_vs_inspection_gain",
    "wrong_region_penalty_sensitivity",
    "family_specific_intervention_accuracy",
    "recurrent_intervention_accuracy",
    "field_intervention_accuracy",
    "schema_intervention_accuracy",
    "trace_ablation_intervention_drop",
    "delay_ood_intervention_accuracy",
    "gain_over_random",
    "gain_over_saliency",
    "gain_over_short_horizon",
    "gain_over_inspect_only",
    "oracle_intervention_score",
    "random_intervention_score",
    "saliency_intervention_score",
    "short_horizon_intervention_score",
    "inspect_only_score",
    "b4_intervention_score",
]


def intervention_region_accuracy(predicted_action: dict, oracle_action: dict, tolerance: int = 0) -> float:
    return 1.0 if int(predicted_action.get("region_id", -1)) == int(oracle_action.get("region_id", -2)) else 0.0


def action_type_accuracy(predicted_action: dict, oracle_action: dict) -> float:
    return 1.0 if str(predicted_action.get("action_type", "")) == str(oracle_action.get("action_type", "")) else 0.0


def trace_guided_intervention_accuracy(predicted_action: dict, oracle_action: dict) -> float:
    return intervention_region_accuracy(predicted_action, oracle_action) * action_type_accuracy(predicted_action, oracle_action)


def outcome_improvement(baseline_outcome: dict, intervened_outcome: dict) -> float:
    return max(0.0, float(baseline_outcome.get("outcome_error", 1.0)) - float(intervened_outcome.get("outcome_error", 1.0)))


def intervention_vs_inspection_gain(intervention_gain: float, inspect_only_gain: float) -> float:
    return float(intervention_gain) - float(inspect_only_gain)


def wrong_region_penalty_sensitivity(correct_value: float, wrong_region_value: float) -> float:
    return max(0.0, float(correct_value) - float(wrong_region_value))


def family_specific_intervention_accuracy(predicted_action: dict, expected_family_action: dict) -> float:
    return trace_guided_intervention_accuracy(predicted_action, expected_family_action)


def delay_ood_intervention_accuracy(results_by_delay: dict[int, float]) -> float:
    return sum(float(value) for value in results_by_delay.values()) / len(results_by_delay) if results_by_delay else 0.0


def b4_intervention_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = {**B4_GATES, **(gates or {})}
    min_gates = [
        "trace_guided_intervention_accuracy",
        "intervention_region_accuracy",
        "action_type_accuracy",
        "outcome_improvement",
        "intervention_vs_inspection_gain",
        "wrong_region_penalty_sensitivity",
        "family_specific_intervention_accuracy",
        "recurrent_intervention_accuracy",
        "field_intervention_accuracy",
        "schema_intervention_accuracy",
        "trace_ablation_intervention_drop",
        "delay_ood_intervention_accuracy",
        "gain_over_random",
        "gain_over_saliency",
        "gain_over_short_horizon",
        "gain_over_inspect_only",
        "oracle_intervention_score",
    ]
    for key in min_gates:
        if float(metrics.get(key, 0.0)) < float(gates[key]):
            return 0.0
    if float(metrics.get("random_intervention_score", 1.0)) > float(gates["random_intervention_score_max"]):
        return 0.0
    weights = {
        "trace_guided_intervention_accuracy": 0.20,
        "outcome_improvement": 0.20,
        "intervention_vs_inspection_gain": 0.15,
        "wrong_region_penalty_sensitivity": 0.15,
        "family_specific_intervention_accuracy": 0.15,
        "trace_ablation_intervention_drop": 0.15,
    }
    return float(sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items()))

