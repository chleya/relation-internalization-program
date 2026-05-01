from src.b52_adaptive_update_env import make_b52_adaptive_update_episode_pair
from src.b52_feedback_stress import evaluate_feedback_content_sensitivity, make_contradictory_feedback, make_delayed_feedback
from src.b52_plan_divergence import adaptive_intervention_plan
from src.models import make_model


def test_feedback_variants_generated():
    model = make_model("recurrent_flow_checkpoint_model")
    bundle = make_b52_adaptive_update_episode_pair({}, 4, "contradictory_feedback")["episode_a"]
    output = adaptive_intervention_plan(model, bundle["model_input"], {})
    assert make_contradictory_feedback(bundle, output, {})["feedback_type"] == "contradictory"
    assert make_delayed_feedback(bundle, output, 3, {})["delay_steps"] == 3


def test_feedback_sensitivity_metric_runs():
    model = make_model("recurrent_flow_checkpoint_model")
    bundle = make_b52_adaptive_update_episode_pair({}, 4, "delayed_feedback")["episode_a"]
    metrics, records = evaluate_feedback_content_sensitivity(model, [bundle], {})
    assert 0.0 <= metrics["feedback_content_sensitivity"] <= 1.0
    assert records
