from __future__ import annotations

from src.b42_action_type_env import make_action_type_specific_episode
from tests.conftest import small_config


def test_action_type_episode_has_required_fields_and_varies():
    config = small_config()
    first = make_action_type_specific_episode(config, 2, "recurrent")
    second = make_action_type_specific_episode(config, 3, "recurrent")
    gt = first["ground_truth"]
    assert gt["required_action_type"]
    assert "intervention_region" in gt
    assert gt["wrong_action_types"]
    assert first["ground_truth"]["required_action_type"] != second["ground_truth"]["required_action_type"]

