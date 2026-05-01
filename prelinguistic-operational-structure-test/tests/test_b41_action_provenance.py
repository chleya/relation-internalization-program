from __future__ import annotations

from src.b41_action_provenance import collect_action_policy_provenance
from src.b4_intervention_env import make_b4_intervention_episode
from src.models import make_model
from tests.conftest import small_config


def test_action_provenance_has_required_flags():
    config = small_config()
    episode = make_b4_intervention_episode(config, 3, "b41", "recurrent")
    records = collect_action_policy_provenance(make_model("recurrent_flow_checkpoint_model"), [episode], config)
    row = records[0]
    assert row["predicted_action_type"]
    assert row["policy_source"]
    assert isinstance(row["shared_action_policy_used"], bool)
    assert isinstance(row["private_trace_action_score_used"], bool)
    assert isinstance(row["oracle_value_used"], bool)

