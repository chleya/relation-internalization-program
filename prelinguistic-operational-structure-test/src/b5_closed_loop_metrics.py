from __future__ import annotations

from statistics import mean


B5_GATES = {
    "inspect_timing_accuracy": 0.70,
    "epistemic_value_alignment": 0.70,
    "trace_update_accuracy": 0.70,
    "post_inspection_intervention_accuracy": 0.70,
    "pragmatic_value_alignment": 0.70,
    "feedback_revision_accuracy": 0.65,
    "closed_loop_gain_over_inspect_always": 0.15,
    "closed_loop_gain_over_intervene_immediately": 0.15,
    "closed_loop_gain_over_random": 0.20,
    "closed_loop_gain_over_saliency": 0.15,
    "closed_loop_gain_over_short_horizon": 0.15,
    "wrong_inspect_penalty_sensitivity": 0.20,
    "wrong_intervention_penalty_sensitivity": 0.20,
    "planning_budget_compliance": 1.00,
    "oracle_closed_loop_score": 0.95,
    "random_closed_loop_score_max": 0.25,
}

B5_SUMMARY_KEYS = [
    "model",
    "seed",
    "inspect_timing_accuracy",
    "epistemic_value_alignment",
    "trace_update_accuracy",
    "trace_uncertainty_reduction",
    "post_inspection_intervention_accuracy",
    "pragmatic_value_alignment",
    "feedback_revision_accuracy",
    "closed_loop_gain_over_inspect_always",
    "closed_loop_gain_over_intervene_immediately",
    "closed_loop_gain_over_random",
    "closed_loop_gain_over_saliency",
    "closed_loop_gain_over_short_horizon",
    "wrong_inspect_penalty_sensitivity",
    "wrong_intervention_penalty_sensitivity",
    "planning_budget_compliance",
    "oracle_closed_loop_score",
    "random_closed_loop_score",
    "b5_closed_loop_score",
]


def inspect_timing_accuracy(predicted: dict, ground_truth: dict) -> float:
    return 1.0 if bool(predicted.get("inspect_chosen", False)) == bool(ground_truth.get("needs_inspection", False)) else 0.0


def epistemic_value_alignment(policy_output: dict, ground_truth: dict) -> float:
    if not bool(ground_truth.get("needs_inspection", False)):
        return 1.0 if not bool(policy_output.get("inspect_chosen", False)) else 0.0
    return 1.0 if int(policy_output.get("predicted_inspect_region", -1)) == int(ground_truth.get("oracle_inspect_region", -2)) else 0.0


def trace_update_accuracy(updated_trace: dict, ground_truth: dict) -> float:
    return 1.0 if int(updated_trace.get("region", -1)) == int(ground_truth.get("trace_after_inspection_region", -2)) else 0.0


def post_inspection_intervention_accuracy(policy_output: dict, ground_truth: dict) -> float:
    action = policy_output.get("intervention_action", {})
    expected = ground_truth.get("oracle_intervention_action", {})
    return 1.0 if same_action(action, expected) else 0.0


def pragmatic_value_alignment(policy_output: dict, ground_truth: dict) -> float:
    return post_inspection_intervention_accuracy(policy_output, ground_truth)


def feedback_revision_accuracy(policy_output: dict, ground_truth: dict) -> float:
    return 1.0 if int(policy_output.get("trace_after_feedback_region", -1)) == int(ground_truth.get("true_trace_region", -2)) else 0.0


def closed_loop_gain_over_baseline(model_score: float, baseline_score: float) -> float:
    return float(model_score) - float(baseline_score)


def wrong_inspect_penalty_sensitivity(correct_value: float, wrong_inspect_value: float) -> float:
    return max(0.0, float(correct_value) - float(wrong_inspect_value))


def wrong_intervention_penalty_sensitivity(correct_value: float, wrong_intervention_value: float) -> float:
    return max(0.0, float(correct_value) - float(wrong_intervention_value))


def b5_closed_loop_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = {**B5_GATES, **(gates or {})}
    min_gates = [
        "inspect_timing_accuracy",
        "epistemic_value_alignment",
        "trace_update_accuracy",
        "post_inspection_intervention_accuracy",
        "pragmatic_value_alignment",
        "feedback_revision_accuracy",
        "closed_loop_gain_over_inspect_always",
        "closed_loop_gain_over_intervene_immediately",
        "closed_loop_gain_over_random",
        "closed_loop_gain_over_saliency",
        "closed_loop_gain_over_short_horizon",
        "wrong_inspect_penalty_sensitivity",
        "wrong_intervention_penalty_sensitivity",
        "planning_budget_compliance",
        "oracle_closed_loop_score",
    ]
    for key in min_gates:
        if float(metrics.get(key, 0.0)) < float(gates[key]):
            return 0.0
    if float(metrics.get("random_closed_loop_score", 1.0)) > float(gates["random_closed_loop_score_max"]):
        return 0.0
    weights = {
        "inspect_timing_accuracy": 0.15,
        "trace_update_accuracy": 0.20,
        "post_inspection_intervention_accuracy": 0.20,
        "feedback_revision_accuracy": 0.15,
        "closed_loop_gain_over_intervene_immediately": 0.15,
        "wrong_intervention_penalty_sensitivity": 0.15,
    }
    return float(sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items()))


def mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def same_action(left: dict, right: dict) -> bool:
    return str(left.get("action_type")) == str(right.get("action_type")) and int(left.get("region_id", -1)) == int(right.get("region_id", -2))
