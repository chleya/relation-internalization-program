from __future__ import annotations

import numpy as np

from src.b3_active_inspection_env import make_b3_active_inspection_episode
from src.b3_information_gain import compute_information_gain_after_inspection, delayed_checkpoint_uncertainty, future_endpoint_error
from tests.conftest import small_config


def test_future_endpoint_error_nonnegative():
    frames = np.zeros((2, 64, 64, 1), dtype=np.float32)
    assert future_endpoint_error(frames, frames, small_config()) >= 0.0


def test_information_gain_is_pre_minus_post_and_finite():
    config = small_config()
    episode = make_b3_active_inspection_episode(config, 4, "b3_trace_guided_inspection", delay=4)
    region = episode["ground_truth"]["oracle_best_inspect_region"]
    gain = compute_information_gain_after_inspection(None, episode, region, config)
    assert gain["absolute_gain"] == gain["pre_inspection_error"] - gain["post_inspection_error"]
    assert gain["absolute_gain"] >= 0.0
    assert np.isfinite(gain["relative_gain"])


def test_delayed_checkpoint_uncertainty_in_range():
    uncertainty = delayed_checkpoint_uncertainty({"inspection_logits": np.asarray([0.0, 2.0, 0.0])}, {}, small_config())
    assert 0.0 <= uncertainty <= 1.0
