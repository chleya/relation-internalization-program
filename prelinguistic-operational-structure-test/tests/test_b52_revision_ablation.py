from src.b52_adaptive_update_env import make_b52_adaptive_update_episode_pair
from src.b52_revision_ablation import (
    ablate_feedback_revision_path,
    ablate_non_revision_control_path,
    ablate_trace_update_path,
    evaluate_revision_specific_ablation,
)
from src.models import make_model


def test_revision_ablation_wrappers_run():
    model = make_model("schema_memory_model")
    assert ablate_trace_update_path(model, {}).ablation_type == "trace_update"
    assert ablate_feedback_revision_path(model, {}).ablation_type == "feedback_revision"
    assert ablate_non_revision_control_path(model, {}).ablation_type == "non_revision_control"


def test_revision_ablation_metrics_finite():
    model = make_model("schema_memory_model")
    bundle = make_b52_adaptive_update_episode_pair({}, 6, "scripted_feedback_trap")["episode_a"]
    metrics, records = evaluate_revision_specific_ablation(model, [bundle], {})
    assert metrics["revision_specific_ablation_drop"] >= 0.0
    assert metrics["non_revision_path_stability"] >= 0.0
    assert records
