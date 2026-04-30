from __future__ import annotations

from src.data import generate_episode

from .conftest import small_config


def test_occlusion_crossing_metadata_exists() -> None:
    episode = generate_episode(small_config(), seed=2, episode_type="occlusion_crossing")
    gt = episode["ground_truth"]
    assert "occlusion_intervals" in gt
    assert "crossing_intervals" in gt
    assert isinstance(gt["occlusion_intervals"], list)
    assert isinstance(gt["crossing_intervals"], list)
