from __future__ import annotations

from src.b21_trace_attacks import make_trace_swap_pair
from src.b21_trace_interventions import (
    apply_matched_non_trace_deletion,
    apply_trace_compression,
    apply_trace_swap,
    apply_true_trace_deletion,
)
from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.model_io import make_model_batch
from src.models import make_model
from tests.conftest import small_config


def b21_small_config():
    config = small_config()
    config["b2"] = {
        "frame_size": 64,
        "grid_size": 8,
        "past_frames": 8,
        "future_frames": 12,
        "delays": [2, 4, 6],
        "heldout_delays": [3, 5, 7],
    }
    return config


def test_true_trace_deletion_does_not_crash():
    config = b21_small_config()
    episode = make_delayed_checkpoint_episode(config, 0, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("field_memory_model")
    result = apply_true_trace_deletion(model, batch, "field_memory")
    assert "applicable" in result


def test_matched_non_trace_deletion_does_not_crash():
    config = b21_small_config()
    episode = make_delayed_checkpoint_episode(config, 1, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("schema_memory_model")
    result = apply_matched_non_trace_deletion(model, batch, "schema_memory")
    assert "applicable" in result


def test_trace_swap_returns_outputs():
    config = b21_small_config()
    a, b = make_trace_swap_pair(config, 2, delay=4)
    batch_a = make_model_batch(a, config)
    batch_b = make_model_batch(b, config)
    model = make_model("recurrent_flow_checkpoint_model")
    out_a, out_b = apply_trace_swap(model, batch_a, batch_b, "recurrent_flow_checkpoint")
    assert out_a.get("applicable") is True
    assert out_b.get("applicable") is True


def test_compression_levels_do_not_crash():
    config = b21_small_config()
    episode = make_delayed_checkpoint_episode(config, 3, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("field_memory_model")
    result = apply_trace_compression(model, batch, 0.5, "field_memory")
    assert result.get("applicable") is True


def test_unsupported_trace_family_returns_not_applicable():
    config = b21_small_config()
    episode = make_delayed_checkpoint_episode(config, 4, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("pixel_predictor")
    result = apply_true_trace_deletion(model, batch, "pixel")
    assert result.get("applicable") is False
