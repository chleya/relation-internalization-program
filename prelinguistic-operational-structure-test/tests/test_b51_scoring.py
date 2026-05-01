from src.b51_closed_loop_degeneracy_audit import b51_closed_loop_audit_score


def passing_metrics():
    return {
        "cross_model_exact_plan_match_rate": 0.0,
        "exact_all_model_same_plan_rate": 0.0,
        "fixed_closed_loop_plan_rate": 0.2,
        "inspect_always_rate": 0.5,
        "intervene_immediately_rate": 0.2,
        "skip_inspect_when_not_needed_rate": 1.0,
        "skip_intervention_when_not_needed_rate": 1.0,
        "decision_diversity_score": 0.75,
        "shared_closed_loop_policy_usage_rate": 0.0,
        "private_trace_closed_loop_usage_rate": 1.0,
        "fallback_usage_rate": 0.0,
        "trace_update_specificity": 1.0,
        "trace_update_ablation_drop": 0.3,
        "trace_update_over_non_trace_ratio": 2.0,
        "feedback_revision_specificity": 1.0,
        "feedback_revision_ablation_drop": 0.3,
        "feedback_revision_over_scripted_ratio": 2.0,
        "random_closed_loop_score": 0.1,
        "oracle_closed_loop_score": 1.0,
        "model_gain_over_inspect_always": 0.2,
        "model_gain_over_intervene_immediately": 0.2,
        "model_gain_over_scripted_update": 0.2,
        "value_leakage_count": 0.0,
        "oracle_plan_usage_rate": 0.0,
        "oracle_trace_update_usage_rate": 0.0,
        "oracle_feedback_revision_usage_rate": 0.0,
        "planning_budget_stress_retention": 1.0,
        "strict_budget_compliance": 1.0,
    }


def test_b51_score_passes_when_gates_pass():
    assert b51_closed_loop_audit_score(passing_metrics(), {}) > 0.0


def test_b51_score_zero_for_plan_overlap():
    metrics = passing_metrics()
    metrics["cross_model_exact_plan_match_rate"] = 1.0
    assert b51_closed_loop_audit_score(metrics, {}) == 0.0


def test_b51_score_zero_for_leakage():
    metrics = passing_metrics()
    metrics["value_leakage_count"] = 1.0
    assert b51_closed_loop_audit_score(metrics, {}) == 0.0
