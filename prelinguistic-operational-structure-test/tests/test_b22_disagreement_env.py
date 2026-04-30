from src.b22_disagreement_env import make_trace_disagreement_episode


def small_config():
    return {"env": {"frame_size": 64, "grid_size": 8, "past_frames": 8, "future_frames": 12}}


def test_trace_disagreement_episode_has_distinct_trace_regions():
    episode = make_trace_disagreement_episode(small_config(), 0, "field")
    gt = episode["ground_truth"]
    assert gt["episode_type"] == "b22_trace_disagreement"
    assert "temporal_trace_region" in gt
    assert "field_trace_region" in gt
    assert "schema_trace_region" in gt
    assert len({gt["temporal_trace_region"], gt["field_trace_region"], gt["schema_trace_region"]}) > 1
    assert gt["causal_family"] == "field"
    assert gt["causal_region"] == gt["field_trace_region"]
