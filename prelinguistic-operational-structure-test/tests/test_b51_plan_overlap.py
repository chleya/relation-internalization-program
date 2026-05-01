from src.b51_plan_overlap import closed_loop_plan_signature, compute_fixed_closed_loop_plan_rate, compute_per_episode_plan_overlap


def record(model="a", episode_id=0, action="do_nothing"):
    return {
        "record_kind": "closed_loop_policy",
        "model": model,
        "episode_id": episode_id,
        "inspect_skipped": 1,
        "predicted_inspect_region": -1,
        "trace_after_inspection_region": 3,
        "predicted_intervention_action_type": action,
        "predicted_intervention_region": 3,
        "trace_after_feedback_region": 3,
    }


def test_closed_loop_plan_signature_is_tuple():
    assert isinstance(closed_loop_plan_signature(record()), tuple)


def test_plan_overlap_metrics_in_range():
    records = [record("a"), record("b"), record("c")]
    metrics = compute_per_episode_plan_overlap(records, {})
    assert 0.0 <= metrics["cross_model_exact_plan_match_rate"] <= 1.0
    assert compute_fixed_closed_loop_plan_rate(records, {})["fixed_closed_loop_plan_rate"] == 1.0
