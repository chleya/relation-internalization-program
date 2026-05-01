from __future__ import annotations

from src.b42_action_type_baselines import fixed_action_baseline_policy, oracle_action_type_baseline_policy, random_action_type_baseline_policy
from src.b42_action_type_env import make_action_type_specific_episode
from src.b42_action_type_metrics import joint_region_action_accuracy
from tests.conftest import small_config


def test_b42_baselines_run_and_oracle_beats_fixed():
    config = small_config()
    episode = make_action_type_specific_episode(config, 17, "field")
    oracle = episode["ground_truth"]["oracle_best_action"]
    fixed = fixed_action_baseline_policy(episode, config, "apply_local_damping")
    random = random_action_type_baseline_policy(episode, config, 0)
    oracle_action = oracle_action_type_baseline_policy(episode, config)
    assert fixed["action_type"]
    assert random["action_type"]
    assert joint_region_action_accuracy(oracle_action, oracle) >= joint_region_action_accuracy(fixed, oracle)

