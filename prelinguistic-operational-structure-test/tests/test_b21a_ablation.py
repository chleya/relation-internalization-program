from __future__ import annotations

from src.b21a_degeneracy_audit import apply_no_trace_ablation, apply_trace_family_ablation, evaluate_trace_family_ablation
from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.model_io import make_model_batch
from src.models import make_model
from tests.conftest import small_config


def b21a_small_config():
    config = small_config()
    config["b2"] = {
        "frame_size": 64,
        "grid_size": 8,
        "past_frames": 8,
        "future_frames": 12,
        "delays": [2, 4, 6],
    }
    return config


def test_trace_family_ablation_does_not_crash():
    config = b21a_small_config()
    episode = make_delayed_checkpoint_episode(config, 0, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("field_memory_model")
    result = apply_trace_family_ablation(model, batch, "field_memory")
    assert result.get("applicable") is True


def test_no_trace_ablation_does_not_crash():
    config = b21a_small_config()
    episode = make_delayed_checkpoint_episode(config, 1, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("schema_memory_model")
    result = apply_no_trace_ablation(model, batch)
    assert result.get("applicable") is True


def test_unsupported_trace_family_returns_not_applicable():
    config = b21a_small_config()
    episode = make_delayed_checkpoint_episode(config, 2, delay=4)
    batch = make_model_batch(episode, config)
    model = make_model("pixel_predictor")
    result = apply_trace_family_ablation(model, batch, "pixel")
    assert result.get("applicable") is False


def test_evaluate_trace_family_ablation_returns_drops():
    config = b21a_small_config()
    episodes = [make_delayed_checkpoint_episode(config, 3, delay=4)]
    model = make_model("recurrent_flow_checkpoint_model")
    result = evaluate_trace_family_ablation(model, episodes, config)
    assert "trace_family_ablation_drop" in result
    assert "no_trace_ablation_drop" in result
