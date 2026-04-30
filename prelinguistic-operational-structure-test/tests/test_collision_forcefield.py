from __future__ import annotations

from src.data import generate_episode

from .conftest import small_config


def test_collision_metadata_exists() -> None:
    episode = generate_episode(small_config(), seed=3, episode_type="collision_bounce")
    gt = episode["ground_truth"]
    assert "collision_time" in gt
    assert "collision_pairs" in gt


def test_forcefield_metadata_exists() -> None:
    episode = generate_episode(small_config(), seed=4, episode_type="forcefield")
    gt = episode["ground_truth"]
    assert "force_field_region" in gt
    assert "critical_inspection_region" in gt
    assert "forcefield_intervals" in gt
    assert "event_points" in gt


def test_forcefield_leaves_past_observable_deviation() -> None:
    episode = generate_episode(small_config(), seed=14, episode_type="forcefield")
    velocities = episode["ground_truth"]["true_velocities"][: small_config()["env"]["past_frames"]]
    accel = velocities[1:] - velocities[:-1]
    assert float(abs(accel).max()) > 0.1
    assert episode["ground_truth"]["forcefield_intervals"]
