from src.b52_adaptive_update_metrics import b52_adaptive_update_score


def passing_metrics():
    return {
        "value_leakage_count": 0.0,
        "oracle_plan_usage_rate": 0.0,
        "oracle_trace_update_usage_rate": 0.0,
        "oracle_feedback_revision_usage_rate": 0.0,
        "cross_model_exact_plan_match_rate": 0.0,
        "exact_all_model_same_plan_rate": 0.0,
        "random_update_score": 0.0,
        "inspection_content_sensitivity": 1.0,
        "inspection_swap_update_change_rate": 1.0,
        "counterfactual_update_switch_rate": 1.0,
        "same_initial_different_info_plan_divergence": 1.0,
        "post_update_plan_divergence": 1.0,
        "post_update_intervention_change_rate": 1.0,
        "model_gain_over_scripted_update": 0.5,
        "update_specificity_over_scripted": 2.0,
        "update_specificity_over_shuffled": 2.0,
        "feedback_content_sensitivity": 1.0,
        "contradictory_feedback_revision_accuracy": 1.0,
        "delayed_feedback_revision_accuracy": 1.0,
        "model_gain_over_scripted_feedback": 0.5,
        "feedback_specificity_over_scripted": 2.0,
        "revision_specific_ablation_drop": 1.0,
        "update_path_ablation_drop": 1.0,
        "feedback_path_ablation_drop": 1.0,
        "non_revision_path_stability": 1.0,
        "oracle_adaptive_update_score": 1.0,
    }


def test_b52_score_zero_on_leakage():
    metrics = passing_metrics()
    metrics["value_leakage_count"] = 1.0
    assert b52_adaptive_update_score(metrics, {}) == 0.0


def test_b52_score_zero_on_low_content_sensitivity():
    metrics = passing_metrics()
    metrics["inspection_content_sensitivity"] = 0.0
    assert b52_adaptive_update_score(metrics, {}) == 0.0


def test_b52_score_zero_on_same_plan_degeneracy():
    metrics = passing_metrics()
    metrics["cross_model_exact_plan_match_rate"] = 1.0
    assert b52_adaptive_update_score(metrics, {}) == 0.0


def test_b52_score_zero_when_scripted_matches():
    metrics = passing_metrics()
    metrics["model_gain_over_scripted_update"] = 0.0
    assert b52_adaptive_update_score(metrics, {}) == 0.0


def test_b52_score_positive_when_all_gates_pass():
    assert b52_adaptive_update_score(passing_metrics(), {}) > 0.0
