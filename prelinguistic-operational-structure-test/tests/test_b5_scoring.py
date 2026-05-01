from src.b5_closed_loop_metrics import b5_closed_loop_score


def passing_metrics():
    return {
        "inspect_timing_accuracy": 1.0,
        "epistemic_value_alignment": 1.0,
        "trace_update_accuracy": 1.0,
        "post_inspection_intervention_accuracy": 1.0,
        "pragmatic_value_alignment": 1.0,
        "feedback_revision_accuracy": 1.0,
        "closed_loop_gain_over_inspect_always": 0.2,
        "closed_loop_gain_over_intervene_immediately": 0.2,
        "closed_loop_gain_over_random": 0.3,
        "closed_loop_gain_over_saliency": 0.2,
        "closed_loop_gain_over_short_horizon": 0.2,
        "wrong_inspect_penalty_sensitivity": 0.3,
        "wrong_intervention_penalty_sensitivity": 0.3,
        "planning_budget_compliance": 1.0,
        "oracle_closed_loop_score": 1.0,
        "random_closed_loop_score": 0.1,
    }


def test_b5_score_passes_when_all_gates_pass():
    assert b5_closed_loop_score(passing_metrics(), {}) > 0.0


def test_b5_score_zero_when_trace_update_fails():
    metrics = passing_metrics()
    metrics["trace_update_accuracy"] = 0.0
    assert b5_closed_loop_score(metrics, {}) == 0.0


def test_b5_score_zero_when_random_baseline_high():
    metrics = passing_metrics()
    metrics["random_closed_loop_score"] = 0.9
    assert b5_closed_loop_score(metrics, {}) == 0.0
