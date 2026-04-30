from __future__ import annotations

from src.b3_active_inspection_baselines import (
    evaluate_b3_baselines,
    oracle_inspection_policy,
    random_inspection_policy,
    saliency_inspection_policy,
    short_horizon_checkpoint_policy,
)
from src.b3_active_inspection_env import make_b3_active_inspection_episode
from tests.conftest import small_config


def test_b3_baseline_policies_run():
    config = small_config()
    episode = make_b3_active_inspection_episode(config, 5, "b3_trace_vs_saliency_conflict", delay=4)
    for policy in [
        random_inspection_policy(episode, config, seed=0),
        saliency_inspection_policy(episode, config),
        short_horizon_checkpoint_policy(episode, config),
        oracle_inspection_policy(episode, config),
    ]:
        assert 0 <= policy["inspect_region"] < 64
        assert policy["policy_source"]


def test_oracle_baseline_is_at_least_random_in_sanity_case():
    config = small_config()
    episodes = [make_b3_active_inspection_episode(config, seed + 10, "b3_trace_guided_inspection", delay=4) for seed in range(6)]
    metrics, records = evaluate_b3_baselines(episodes, config, seed=0)
    assert metrics["oracle_inspection_score"] >= metrics["random_inspection_score"]
    assert metrics["oracle_information_gain"] >= metrics["random_information_gain"]
    assert len(records) == 4 * len(episodes)
