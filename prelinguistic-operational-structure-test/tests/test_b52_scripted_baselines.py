from src.b52_adaptive_update_env import make_b52_adaptive_update_episode_pair
from src.b52_scripted_baselines import (
    evaluate_feedback_vs_scripted_baseline,
    evaluate_update_vs_scripted_baseline,
    scripted_feedback_revision_baseline,
    scripted_trace_update_baseline,
)
from src.models import make_model


def test_scripted_baselines_run_without_ground_truth():
    bundle = make_b52_adaptive_update_episode_pair({}, 5, "scripted_update_trap")["episode_a"]
    model_input = bundle["model_input"]
    update = scripted_trace_update_baseline(model_input, model_input["inspection_observation"], {})
    feedback = scripted_feedback_revision_baseline(model_input, model_input["previous_consequence"], {})
    assert update["source"] == "scripted_previous_trace_update"
    assert feedback["source"] == "scripted_previous_trace_feedback"


def test_gain_over_scripted_computed():
    model = make_model("field_memory_model")
    bundle = make_b52_adaptive_update_episode_pair({}, 5, "scripted_feedback_trap")["episode_a"]
    update_metrics, _ = evaluate_update_vs_scripted_baseline(model, [bundle], {})
    feedback_metrics, _ = evaluate_feedback_vs_scripted_baseline(model, [bundle], {})
    assert "model_gain_over_scripted_update" in update_metrics
    assert "model_gain_over_scripted_feedback" in feedback_metrics
