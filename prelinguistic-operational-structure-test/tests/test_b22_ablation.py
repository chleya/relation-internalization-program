from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.b22_selector_ablation import (
    ablate_field_trace,
    ablate_recurrent_trace,
    ablate_schema_trace,
    ablate_shared_selector,
)
from src.model_io import make_model_batch
from src.models import make_model


def small_config():
    return {"env": {"frame_size": 64, "grid_size": 8, "past_frames": 8, "future_frames": 12}}


def test_source_specific_ablations_do_not_crash():
    config = small_config()
    batch = make_model_batch(make_delayed_checkpoint_episode(config, 0, delay=4), config)
    assert ablate_recurrent_trace(make_model("recurrent_flow_checkpoint_model"), batch)["applicable"] is True
    assert ablate_field_trace(make_model("field_memory_model"), batch)["applicable"] is True
    assert ablate_schema_trace(make_model("schema_memory_model"), batch)["applicable"] is True


def test_shared_selector_ablation_returns_output_or_unsupported():
    config = small_config()
    batch = make_model_batch(make_delayed_checkpoint_episode(config, 1, delay=4), config)
    result = ablate_shared_selector(make_model("field_memory_model"), batch)
    assert result["applicable"] is True
    assert "output" in result


def test_unsupported_ablation_returns_not_applicable():
    config = small_config()
    batch = make_model_batch(make_delayed_checkpoint_episode(config, 2, delay=4), config)
    result = ablate_recurrent_trace(make_model("field_memory_model"), batch)
    assert result["applicable"] is False
