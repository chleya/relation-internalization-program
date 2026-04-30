from __future__ import annotations

from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.b2_delayed_interventions import (
    apply_causal_trace_intervention,
    apply_non_trace_control_intervention,
    evaluate_causal_trace_intervention,
)
from src.model_io import make_model_batch
from src.models import make_model
from tests.conftest import small_config


def b2_small_config():
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


def test_causal_trace_intervention_does_not_crash():
    config = b2_small_config()
    episode = make_delayed_checkpoint_episode(config, 0, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("field_memory_model")
    result = apply_causal_trace_intervention(model, batch, "force_trace_mask")
    assert "applicable" in result


def test_non_trace_control_intervention_does_not_crash():
    config = b2_small_config()
    episode = make_delayed_checkpoint_episode(config, 1, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("schema_memory_model")
    result = apply_non_trace_control_intervention(model, batch)
    assert "applicable" in result


def test_unsupported_model_returns_not_applicable():
    config = b2_small_config()
    episode = make_delayed_checkpoint_episode(config, 2, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("pixel_predictor")
    result = apply_causal_trace_intervention(model, batch, "memory_trace_zero")
    assert result.get("applicable") is False


def test_evaluate_causal_trace_intervention_returns_keys():
    config = b2_small_config()
    episodes = [make_delayed_checkpoint_episode(config, 3, delay=4)]
    model = make_model("recurrent_flow_checkpoint_model")
    metrics = evaluate_causal_trace_intervention(model, episodes, config)
    assert "causal_trace_intervention_drop" in metrics
    assert "non_trace_stability" in metrics
    assert "delayed_endpoint_shift" in metrics
