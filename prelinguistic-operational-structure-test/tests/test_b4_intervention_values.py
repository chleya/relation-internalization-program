from __future__ import annotations

import math

from src.b4_intervention_env import apply_intervention_action, make_b4_intervention_episode
from src.b4_intervention_values import compute_intervention_value, compute_oracle_best_intervention, compute_wrong_region_penalty, wrong_region_action_for
from tests.conftest import small_config


def test_intervention_value_and_oracle_are_finite():
    config = small_config()
    episode = make_b4_intervention_episode(config, 13, "b4_intervention", "field")
    oracle = compute_oracle_best_intervention(episode, config)
    value = compute_intervention_value(episode, oracle, config)
    assert math.isfinite(value)
    assert oracle["action_type"] == episode["ground_truth"]["oracle_best_action"]["action_type"]
    assert oracle["region_id"] == episode["ground_truth"]["oracle_best_action"]["region_id"]


def test_wrong_region_penalty_nonnegative_and_inspect_only_no_dynamics():
    config = small_config()
    episode = make_b4_intervention_episode(config, 17, "b4_intervention", "recurrent")
    oracle = episode["ground_truth"]["oracle_best_action"]
    wrong = wrong_region_action_for(episode, oracle, config)
    assert compute_wrong_region_penalty(episode, oracle, wrong, config) >= 0.0
    inspected = apply_intervention_action(episode, {"action_type": "inspect_only", "region_id": oracle["region_id"], "strength": 1.0}, config)
    assert inspected["intervention"]["dynamics_changed"] is False

