from __future__ import annotations

from src.b3_active_inspection_env import make_b3_active_inspection_episode
from src.b3_inspection_policy import evaluate_trace_guided_policy, trace_guided_inspection_policy
from src.models import make_model
from tests.conftest import small_config


def test_trace_guided_inspection_policy_returns_valid_region():
    config = small_config()
    episode = make_b3_active_inspection_episode(config, 3, "b3_trace_guided_inspection", delay=4)
    policy = trace_guided_inspection_policy(make_model("recurrent_flow_checkpoint_model"), episode, config)
    assert 0 <= policy["inspect_region"] < 64
    assert policy["policy_source"]
    assert policy["trace_family"]
    assert policy["provenance"]["shared_selector_used"] is False
    assert policy["provenance"]["model_private_score_used"] is True


def test_evaluate_trace_guided_policy_returns_core_metrics():
    config = small_config()
    episodes = [make_b3_active_inspection_episode(config, seed, "b3_trace_guided_inspection", delay=4) for seed in range(2)]
    metrics, records = evaluate_trace_guided_policy(make_model("field_memory_model"), episodes, config)
    assert 0.0 <= metrics["trace_guided_inspection_accuracy"] <= 1.0
    assert 0.0 <= metrics["trace_vs_saliency_rejection"] <= 1.0
    assert len(records) == len(episodes)
