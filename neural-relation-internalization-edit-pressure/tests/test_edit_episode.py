from __future__ import annotations

from src.data import generate_edit_episode


def test_edit_episode_contains_support_query_edit_and_targets() -> None:
    episode = generate_edit_episode(seed=0)
    assert episode["support"]
    assert "query" in episode
    assert "edit" in episode
    assert episode["target_before"] == "food"
    assert episode["target_after"] == "poison"
