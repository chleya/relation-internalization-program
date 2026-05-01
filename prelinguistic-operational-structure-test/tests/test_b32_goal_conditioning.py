from __future__ import annotations

from src.b32_goal_conditioning import attach_goal_code, expected_region_for_goal, goal_family_from_code
from src.b32_mechanism_inspection_env import make_family_specific_inspection_episode
from tests.conftest import small_config


def test_goal_code_attaches_and_decodes():
    config = small_config()
    episode = make_family_specific_inspection_episode(config, 19, "recurrent_goal")
    updated = attach_goal_code(episode, "field_goal", config)
    assert updated["goal_code"] == [0.0, 1.0, 0.0]
    assert goal_family_from_code(updated["goal_code"]) == "field_goal"
    assert updated["ground_truth"]["goal_family"] == "field_goal"


def test_expected_region_for_goal_returns_family_target():
    episode = make_family_specific_inspection_episode(small_config(), 23, "schema_goal")
    gt = episode["ground_truth"]
    assert expected_region_for_goal(episode, "recurrent_goal") == gt["recurrent_inspect_region"]
    assert expected_region_for_goal(episode, "field_goal") == gt["field_inspect_region"]
    assert expected_region_for_goal(episode, "schema_goal") == gt["schema_inspect_region"]
