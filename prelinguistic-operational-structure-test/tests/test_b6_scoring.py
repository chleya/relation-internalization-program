from src.b6_risk_metrics import b6_risk_constrained_score


def passing_metrics():
    return {
        "value_leakage_count": 0.0,
        "oracle_actionability_usage_rate": 0.0,
        "oracle_risk_value_usage_rate": 0.0,
        "random_risk_constrained_score": 0.0,
        "actionability_mask_accuracy": 1.0,
        "inspectable_decision_accuracy": 1.0,
        "direct_intervention_accuracy": 1.0,
        "indirect_intervention_accuracy": 1.0,
        "unsafe_action_rejection_rate": 1.0,
        "irreversible_action_rejection_rate": 1.0,
        "costly_action_avoidance_accuracy": 1.0,
        "abstain_when_required_accuracy": 1.0,
        "act_when_safe_and_needed_accuracy": 1.0,
        "wrong_actionability_penalty_sensitivity": 0.7,
        "risk_adjusted_value_alignment": 1.0,
        "cost_sensitive_planning_accuracy": 1.0,
        "risk_aware_feedback_revision_accuracy": 1.0,
        "unsafe_feedback_correction_rate": 1.0,
        "irreversible_feedback_correction_rate": 1.0,
        "gain_over_random": 1.0,
        "gain_over_saliency": 1.0,
        "gain_over_short_horizon": 1.0,
        "gain_over_risk_blind": 1.0,
        "gain_over_always_act": 1.0,
        "gain_over_always_abstain": 1.0,
        "oracle_risk_constrained_score": 1.0,
    }


def test_b6_score_zero_on_unsafe_failure():
    metrics = passing_metrics()
    metrics["unsafe_action_rejection_rate"] = 0.0
    assert b6_risk_constrained_score(metrics, {}) == 0.0


def test_b6_score_zero_on_irreversible_failure():
    metrics = passing_metrics()
    metrics["irreversible_action_rejection_rate"] = 0.0
    assert b6_risk_constrained_score(metrics, {}) == 0.0


def test_b6_score_zero_on_risk_blind_match():
    metrics = passing_metrics()
    metrics["gain_over_risk_blind"] = 0.0
    assert b6_risk_constrained_score(metrics, {}) == 0.0


def test_b6_score_zero_on_leakage():
    metrics = passing_metrics()
    metrics["value_leakage_count"] = 1.0
    assert b6_risk_constrained_score(metrics, {}) == 0.0


def test_b6_score_positive_when_all_gates_pass():
    assert b6_risk_constrained_score(passing_metrics(), {}) > 0.0
