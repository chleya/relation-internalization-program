from __future__ import annotations

from src.b2_delayed_env import make_delay_ood_episode, make_delayed_checkpoint_episode, make_multi_delay_checkpoint_episode
from tests.conftest import small_config


def b2_small_config():
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


def test_make_delayed_checkpoint_episode_metadata():
    episode = make_delayed_checkpoint_episode(b2_small_config(), 0, delay=4)
    gt = episode["ground_truth"]
    assert gt["episode_type"] == "b2_delayed_checkpoint"
    assert gt["delay"] == 4
    assert "true_delayed_checkpoint_region" in gt
    assert "early_saliency_region" in gt
    assert "delayed_causal_time" in gt
    assert gt["critical_inspection_region"] == gt["true_delayed_checkpoint_region"]


def test_make_multi_delay_checkpoint_episode_metadata():
    episode = make_multi_delay_checkpoint_episode(b2_small_config(), 1, [2, 4, 6])
    gt = episode["ground_truth"]
    assert gt["episode_type"] == "b2_multi_delay_checkpoint"
    assert gt["candidate_delays"] == [2, 4, 6]
    assert gt["best_delay"] in gt["candidate_delays"]
    assert gt["best_region"] in gt["candidate_regions"]


def test_make_delay_ood_episode_uses_heldout_delay():
    episode = make_delay_ood_episode(b2_small_config(), 2, heldout_delay=5)
    gt = episode["ground_truth"]
    assert gt["episode_type"] == "b2_delay_ood"
    assert gt["heldout_delay"] == 5
    assert gt["ood_type"] == "heldout_delay"
