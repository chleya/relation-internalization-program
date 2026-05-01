from src.b5_closed_loop_env import inspect_region, make_b5_closed_loop_episode


def test_b5_episode_has_closed_loop_fields():
    episode = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    gt = episode["ground_truth"]
    assert "oracle_closed_loop_plan" in gt
    assert "oracle_inspect_region" in gt
    assert "oracle_intervention_action" in gt
    assert "trace_state" in episode


def test_b5_inspection_reveals_trace_at_oracle_region():
    episode = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    result = inspect_region(episode, episode["ground_truth"]["oracle_inspect_region"], {})
    assert result["reveals_trace"] is True
    assert result["observed_trace_region"] == episode["ground_truth"]["true_trace_region"]
