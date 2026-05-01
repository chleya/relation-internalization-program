from __future__ import annotations

from src.b32_mechanism_inspection_env import make_family_specific_inspection_episode
from src.b32_mechanism_policy import mechanism_conditioned_inspection_policy
from src.models import make_model
from tests.conftest import small_config


def test_mechanism_policy_returns_valid_region_and_provenance():
    config = small_config()
    episode = make_family_specific_inspection_episode(config, 29, "recurrent_goal")
    model = make_model("recurrent_flow_checkpoint_model")
    policy = mechanism_conditioned_inspection_policy(model, episode, [1.0, 0.0, 0.0], config)
    assert 0 <= int(policy["inspect_region"]) < 64
    assert policy["goal_family"] == "recurrent_goal"
    assert policy["trace_family"] == "recurrent"
    assert policy["provenance"]["oracle_region_used"] is False
    assert policy["provenance"]["shared_inspection_policy_used"] is False
