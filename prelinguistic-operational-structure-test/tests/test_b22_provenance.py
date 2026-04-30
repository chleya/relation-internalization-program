from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.b22_provenance import collect_trace_provenance, summarize_trace_provenance
from src.models import make_model


def small_config():
    return {"env": {"frame_size": 64, "grid_size": 8, "past_frames": 8, "future_frames": 12}}


def test_provenance_record_has_required_fields():
    config = small_config()
    episode = make_delayed_checkpoint_episode(config, 0, delay=4)
    model = make_model("recurrent_flow_checkpoint_model")
    rows = collect_trace_provenance(model, [episode], config)
    assert rows[0]["selected_region"] >= 0
    assert rows[0]["source_module"]
    assert rows[0]["source_trace_family"]
    assert isinstance(rows[0]["shared_selector_used"], bool)
    assert isinstance(rows[0]["fallback_used"], bool)


def test_provenance_summary_detects_shared_selector():
    config = small_config()
    episode = make_delayed_checkpoint_episode(config, 1, delay=4)
    rows = collect_trace_provenance(make_model("field_memory_model"), [episode], config)
    summary = summarize_trace_provenance(rows, config)
    assert summary["shared_selector_usage_rate"] == 1.0
