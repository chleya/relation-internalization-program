from __future__ import annotations

from src.b4_intervention_baselines import (
    inspect_only_policy,
    oracle_intervention_policy,
    random_intervention_policy,
    saliency_intervention_policy,
    short_horizon_intervention_policy,
)
from src.b4_intervention_env import make_b4_intervention_episode
from src.b4_intervention_metrics import trace_guided_intervention_accuracy
from tests.conftest import small_config


def test_b4_baselines_run_and_oracle_beats_random_sanity():
    config = small_config()
    episode = make_b4_intervention_episode(config, 19, "b4_intervention", "schema")
    oracle = episode["ground_truth"]["oracle_best_action"]
    random_action = random_intervention_policy(episode, config, 0)
    actions = [
        random_action,
        saliency_intervention_policy(episode, config),
        short_horizon_intervention_policy(episode, config),
        inspect_only_policy(episode, config),
        oracle_intervention_policy(episode, config),
    ]
    assert all("action_type" in action and "region_id" in action for action in actions)
    assert trace_guided_intervention_accuracy(oracle_intervention_policy(episode, config), oracle) >= trace_guided_intervention_accuracy(random_action, oracle)

