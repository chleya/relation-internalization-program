from src.b6_risk_env import make_b6_risk_constrained_episode
from src.b6_risk_feedback_revision import evaluate_risk_aware_feedback_revision, revise_trace_after_risk_feedback
from src.models import make_model


def test_risk_feedback_revision_changes_future_bias():
    bundle = make_b6_risk_constrained_episode({}, 1, "unsafe_direct_intervention")
    target = bundle["evaluator_ground_truth"]["target_region"]
    mask = bundle["model_input"]["actionability_mask"]
    revised = revise_trace_after_risk_feedback(object(), {"region": target}, {"region_id": target, "unsafe_feedback": True}, mask, {})
    assert revised["risk_blocked"] is True
    assert revised["future_action_bias"] == "abstain"


def test_risk_feedback_revision_metrics_run():
    model = make_model("schema_memory_model")
    bundle = make_b6_risk_constrained_episode({}, 1, "irreversible_action_trap")
    metrics, records = evaluate_risk_aware_feedback_revision(model, [bundle], {})
    assert metrics["risk_aware_feedback_revision_accuracy"] >= 0.0
    assert records
