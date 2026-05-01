from __future__ import annotations

from src.b4_action_space import is_valid_action
from src.b42_action_type_env import make_action_type_specific_episode
from src.b42_action_type_policy import action_type_disambiguating_policy
from src.models import make_model
from tests.conftest import small_config


def test_action_type_policy_returns_valid_action_and_no_oracle_value():
    config = small_config()
    episode = make_action_type_specific_episode(config, 11, "schema")
    policy = action_type_disambiguating_policy(make_model("schema_memory_model"), episode, config)
    assert is_valid_action(policy["action"], config)
    assert policy["action_type_scores"]
    assert policy["provenance"]["oracle_value_used"] is False

