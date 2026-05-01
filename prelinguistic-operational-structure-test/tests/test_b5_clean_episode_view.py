import pytest

from src.b5_clean_episode_view import (
    assert_model_input_is_sanitized,
    find_forbidden_key_paths,
    split_b5_episode_for_clean_run,
)
from src.b5_closed_loop_env import make_b5_closed_loop_episode


def test_split_clean_episode_separates_views():
    raw = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    bundle = split_b5_episode_for_clean_run(raw, {}, episode_id=7)
    assert set(bundle) == {"model_input", "evaluator_ground_truth", "oracle_baseline_view", "metadata"}
    assert "ground_truth" not in bundle["model_input"]
    assert "oracle_closed_loop_plan" in bundle["oracle_baseline_view"]
    assert "oracle_intervention_action" in bundle["evaluator_ground_truth"]
    assert_model_input_is_sanitized(bundle["model_input"], {})


def test_forbidden_paths_are_recursive():
    obj = {"safe": [{"ground_truth": {"oracle_value": 1.0}}]}
    paths = find_forbidden_key_paths(obj, {"ground_truth", "oracle_value"})
    assert "safe[0].ground_truth" in paths
    assert "safe[0].ground_truth.oracle_value" in paths


def test_model_input_rejects_forbidden_key():
    with pytest.raises(AssertionError):
        assert_model_input_is_sanitized({"episode_id": 1, "ground_truth": {}}, {})
