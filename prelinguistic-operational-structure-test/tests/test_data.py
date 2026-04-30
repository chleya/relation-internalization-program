from __future__ import annotations

from src.data import generate_dataset, generate_episode, make_ood_dataset

from .conftest import small_config


def test_generate_episode_has_required_keys() -> None:
    episode = generate_episode(small_config(), seed=0, episode_type="forcefield")
    assert "past_frames" in episode
    assert "future_frames" in episode
    assert "ground_truth" in episode
    assert "force_field_region" in episode["ground_truth"]


def test_generate_dataset_respects_runtime_cap() -> None:
    dataset = generate_dataset(small_config(), "test", seed=0)
    assert len(dataset) == 4


def test_make_ood_dataset() -> None:
    dataset = make_ood_dataset(small_config(), "new_speed", seed=0)
    assert len(dataset) == 4
