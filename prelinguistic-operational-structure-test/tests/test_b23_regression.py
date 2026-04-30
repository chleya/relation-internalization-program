from src.b23_selector_validation import b23_private_selector_score, regression_rows_for_model


GATES = {
    "shared_selector_usage_rate": 0.0,
    "fallback_usage_rate": 0.05,
    "model_private_score_usage_rate": 0.95,
    "b2_delayed_score_min": 0.70,
    "b21_trace_hardening_score_min": 0.70,
    "cross_model_exact_prediction_match_rate_max": 0.70,
    "disagreement_episode_divergence_min": 0.50,
    "trace_family_specificity_min": 0.70,
    "model_private_trace_drop_min": 0.20,
    "shared_selector_ablation_drop_max": 0.05,
    "selector_free_retention_min": 0.60,
}


def passing_metrics():
    return {
        "shared_selector_usage_rate": 0.0,
        "fallback_usage_rate": 0.0,
        "model_private_score_usage_rate": 1.0,
        "b2_delayed_score": 0.90,
        "b21_trace_hardening_score": 0.90,
        "cross_model_exact_prediction_match_rate": 0.40,
        "disagreement_episode_divergence": 0.80,
        "trace_family_specificity": 0.80,
        "model_private_trace_drop": 0.30,
        "shared_selector_ablation_drop": 0.0,
        "selector_free_retention": 1.0,
    }


def test_any_core_gate_fail_forces_zero_score():
    metrics = passing_metrics()
    metrics["shared_selector_usage_rate"] = 0.1
    assert b23_private_selector_score(metrics, GATES) == 0.0


def test_all_core_gates_pass_returns_positive_score():
    assert b23_private_selector_score(passing_metrics(), GATES) > 0.0


def test_regression_matrix_has_core_stage_rows():
    rows = regression_rows_for_model("field_memory_model", 0, passing_metrics(), GATES)
    stages = {row["stage"] for row in rows}
    assert {"B2", "B21", "B21a", "B22"}.issubset(stages)
