from src.b22_selector_disentanglement import b22_disentanglement_score


GATES = {
    "shared_selector_usage_rate": 0.05,
    "max_cross_model_exact_prediction_match_rate": 0.80,
    "min_trace_family_specificity": 0.70,
    "min_model_private_trace_drop": 0.20,
    "max_shared_selector_ablation_advantage": 0.00,
    "min_disagreement_episode_divergence": 0.50,
    "min_selector_free_retention": 0.60,
}


def passing_metrics():
    return {
        "shared_selector_usage_rate": 0.0,
        "cross_model_exact_prediction_match_rate": 0.50,
        "trace_family_specificity": 0.75,
        "model_private_trace_drop": 0.30,
        "shared_selector_ablation_advantage": -0.05,
        "disagreement_episode_divergence": 0.60,
        "selector_free_retention": 0.80,
        "private_trace_retention_after_shared_ablation": 0.70,
    }


def test_shared_selector_usage_too_high_forces_zero():
    metrics = passing_metrics()
    metrics["shared_selector_usage_rate"] = 0.50
    assert b22_disentanglement_score(metrics, GATES) == 0.0


def test_cross_model_prediction_match_too_high_forces_zero():
    metrics = passing_metrics()
    metrics["cross_model_exact_prediction_match_rate"] = 1.0
    assert b22_disentanglement_score(metrics, GATES) == 0.0


def test_trace_family_specificity_too_low_forces_zero():
    metrics = passing_metrics()
    metrics["trace_family_specificity"] = 0.10
    assert b22_disentanglement_score(metrics, GATES) == 0.0


def test_model_private_trace_drop_too_low_forces_zero():
    metrics = passing_metrics()
    metrics["model_private_trace_drop"] = 0.0
    assert b22_disentanglement_score(metrics, GATES) == 0.0


def test_all_gates_pass_returns_positive_score():
    assert b22_disentanglement_score(passing_metrics(), GATES) > 0.0
