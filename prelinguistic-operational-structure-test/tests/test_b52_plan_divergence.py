from src.b52_adaptive_update_env import make_b52_adaptive_update_episode_pair
from src.b52_plan_divergence import adaptive_intervention_plan, closed_loop_plan_signature_after_update, evaluate_same_initial_different_info_plan_divergence
from src.models import make_model


def test_plan_signature_is_tuple():
    model = make_model("schema_memory_model")
    bundle = make_b52_adaptive_update_episode_pair({}, 3, "same_initial_different_inspection")["episode_a"]
    signature = closed_loop_plan_signature_after_update(adaptive_intervention_plan(model, bundle["model_input"], {}))
    assert isinstance(signature, tuple)


def test_same_initial_different_info_divergence_metric_in_range():
    model = make_model("schema_memory_model")
    pair = make_b52_adaptive_update_episode_pair({}, 3, "same_initial_different_inspection")
    metrics, records = evaluate_same_initial_different_info_plan_divergence(model, [pair], {})
    assert 0.0 <= metrics["same_initial_different_info_plan_divergence"] <= 1.0
    assert records
