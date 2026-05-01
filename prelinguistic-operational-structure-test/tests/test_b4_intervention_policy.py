from __future__ import annotations

from src.b4_action_space import is_valid_action
from src.b4_intervention_env import make_b4_intervention_episode
from src.b4_intervention_policy import trace_guided_intervention_policy
from src.models import make_model
from tests.conftest import small_config


def test_trace_guided_policy_returns_valid_action_and_provenance():
    config = small_config()
    episode = make_b4_intervention_episode(config, 11, "b4_intervention", "schema")
    model = make_model("schema_memory_model")
    policy = trace_guided_intervention_policy(model, episode, config)
    assert is_valid_action(policy["action"], config)
    assert policy["trace_family"] == "schema"
    assert policy["provenance"]["oracle_best_action_used"] is False
    assert policy["provenance"]["shared_action_policy_used"] is False

