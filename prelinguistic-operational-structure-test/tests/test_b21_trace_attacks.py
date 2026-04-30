from __future__ import annotations

from src.b21_trace_attacks import (
    evaluate_multi_source_trace_conflict,
    make_false_delayed_trace_episode,
    make_multi_source_trace_conflict_episode,
    make_noisy_delayed_trace_episode,
    make_trace_length_extrapolation_dataset,
    make_trace_swap_pair,
)
from src.models import make_model
from tests.conftest import small_config


def b21_small_config():
    config = small_config()
    config["b2"] = {
        "n_train": 3,
        "n_test": 3,
        "n_ood": 3,
        "frame_size": 64,
        "grid_size": 8,
        "past_frames": 8,
        "future_frames": 12,
        "delays": [2, 4, 6],
        "heldout_delays": [3, 5, 7],
    }
    return config


def test_false_delayed_trace_episode_has_true_and_false_regions():
    episode = make_false_delayed_trace_episode(b21_small_config(), 0, delay=4)
    gt = episode["ground_truth"]
    assert gt["episode_type"] == "b21_false_delayed_trace"
    assert "true_trace_region" in gt
    assert "false_trace_region" in gt
    assert gt["true_trace_region"] != gt["false_trace_region"]


def test_trace_swap_pair_returns_compatible_episodes():
    a, b = make_trace_swap_pair(b21_small_config(), 1, delay=4)
    assert a["ground_truth"]["episode_type"] == "b21_trace_swap"
    assert b["ground_truth"]["episode_type"] == "b21_trace_swap"
    assert a["past_frames"].shape == b["past_frames"].shape


def test_noisy_delayed_trace_episode_includes_noise_level():
    episode = make_noisy_delayed_trace_episode(b21_small_config(), 2, delay=4, noise_level=0.25)
    assert episode["ground_truth"]["episode_type"] == "b21_noisy_delayed_trace"
    assert episode["ground_truth"]["noise_level"] == 0.25


def test_trace_length_extrapolation_dataset_uses_extrapolation_delays():
    episodes = make_trace_length_extrapolation_dataset(b21_small_config(), 3, delays=[8, 10], n=4)
    assert len(episodes) == 4
    assert {episode["ground_truth"]["delay"] for episode in episodes} == {8, 10}


def test_multi_source_conflict_evaluator_returns_metric():
    config = b21_small_config()
    episode = make_multi_source_trace_conflict_episode(config, 4, "trace_conflict")
    metrics, records = evaluate_multi_source_trace_conflict(make_model("field_memory_model"), [episode], config)
    assert "multi_source_conflict_resolution" in metrics
    assert len(records) == 1
