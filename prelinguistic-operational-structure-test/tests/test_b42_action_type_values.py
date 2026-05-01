from __future__ import annotations

from src.b42_action_type_env import allowed_action_types, make_action_type_specific_episode
from src.b42_action_type_values import compute_action_type_value, compute_action_type_value_table, compute_correct_region_wrong_action_penalty
from tests.conftest import small_config


def test_correct_action_value_beats_wrong_action():
    config = small_config()
    episode = make_action_type_specific_episode(config, 5, "field")
    oracle = episode["ground_truth"]["oracle_best_action"]
    wrong = episode["ground_truth"]["wrong_action_types"][0]
    assert compute_action_type_value(episode, oracle["action_type"], oracle["region_id"], config) > compute_action_type_value(episode, wrong, oracle["region_id"], config)
    assert compute_correct_region_wrong_action_penalty(episode, config)["correct_region_wrong_action_penalty"] >= 0.0


def test_value_table_contains_allowed_action_types():
    config = small_config()
    episode = make_action_type_specific_episode(config, 7, "schema")
    table = compute_action_type_value_table(episode, config)
    action_types = {key[0] for key in table}
    assert set(allowed_action_types(config)).issubset(action_types)

