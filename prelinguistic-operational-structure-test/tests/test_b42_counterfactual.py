from __future__ import annotations

import math

from src.b42_action_type_env import allowed_action_types, make_action_type_specific_episode
from src.b42_counterfactual import compute_action_type_counterfactual_sensitivity, run_action_type_counterfactual
from tests.conftest import small_config


def test_counterfactual_evaluates_all_action_types_and_is_sensitive():
    config = small_config()
    episode = make_action_type_specific_episode(config, 13, "recurrent")
    oracle = episode["ground_truth"]["oracle_best_action"]
    results = run_action_type_counterfactual(episode, oracle["region_id"], config)
    assert set(results) == set(allowed_action_types(config))
    sensitivity = compute_action_type_counterfactual_sensitivity(results, oracle["action_type"])
    assert math.isfinite(sensitivity)
    assert results[oracle["action_type"]] > max(value for action, value in results.items() if action != oracle["action_type"])

