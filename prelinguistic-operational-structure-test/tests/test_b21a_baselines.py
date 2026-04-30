from __future__ import annotations

from src.b21a_audit_baselines import OracleTraceBaseline, RandomTraceBaseline, evaluate_oracle_trace_baseline, evaluate_random_trace_baseline
from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.model_io import make_model_batch
from tests.conftest import small_config


def b21a_small_config():
    config = small_config()
    config["b2"] = {
        "frame_size": 64,
        "grid_size": 8,
        "past_frames": 8,
        "future_frames": 12,
        "delays": [2, 4, 6],
    }
    return config


def test_random_trace_baseline_runs():
    config = b21a_small_config()
    episode = make_delayed_checkpoint_episode(config, 0, delay=4)
    batch = make_model_batch(episode, config)
    output = RandomTraceBaseline(seed=0).forward(batch)
    assert "inspection_logits" in output


def test_oracle_trace_baseline_runs():
    config = b21a_small_config()
    episode = make_delayed_checkpoint_episode(config, 1, delay=4)
    batch = make_model_batch(episode, config)
    batch["oracle_trace_region"] = episode["ground_truth"]["true_delayed_checkpoint_region"]
    output = OracleTraceBaseline().forward(batch)
    assert "inspection_logits" in output


def test_random_score_not_above_oracle_score_in_sanity_case():
    config = b21a_small_config()
    episodes = [make_delayed_checkpoint_episode(config, idx, delay=4) for idx in range(4)]
    random_score = evaluate_random_trace_baseline(episodes, config, seed=2)["random_b21_score"]
    oracle_score = evaluate_oracle_trace_baseline(episodes, config)["oracle_b21_score"]
    assert random_score <= oracle_score
