from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.b23_selector_provenance import collect_b23_provenance, summarize_b23_provenance
from src.models import make_model


def small_config():
    return {"env": {"frame_size": 64, "grid_size": 8, "past_frames": 8, "future_frames": 12}}


def test_b23_provenance_has_private_trace_family():
    config = small_config()
    episode = make_delayed_checkpoint_episode(config, 0, delay=4)
    rows = collect_b23_provenance(make_model("schema_memory_model"), [episode], config)
    assert rows[0]["source_trace_family"] in {"recurrent_memory", "field_trace", "schema_memory"}
    assert rows[0]["shared_selector_used"] is False


def test_b23_provenance_summary_rates():
    config = small_config()
    episode = make_delayed_checkpoint_episode(config, 1, delay=4)
    rows = collect_b23_provenance(make_model("recurrent_flow_checkpoint_model"), [episode], config)
    summary = summarize_b23_provenance(rows)
    assert summary["shared_selector_usage_rate"] == 0.0
    assert summary["model_private_score_usage_rate"] == 1.0
