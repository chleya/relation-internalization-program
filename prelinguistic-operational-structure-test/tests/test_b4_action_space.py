from __future__ import annotations

from src.b4_action_space import ACTION_TYPES, action_cost, enumerate_candidate_actions, is_valid_action
from tests.conftest import small_config


def test_action_space_includes_required_actions():
    assert "do_nothing" in ACTION_TYPES
    assert "inspect_only" in ACTION_TYPES
    assert "apply_local_damping" in ACTION_TYPES
    assert "apply_local_push" in ACTION_TYPES
    assert "block_force_region" in ACTION_TYPES
    assert "stabilize_trace_region" in ACTION_TYPES


def test_enumerate_candidate_actions_returns_valid_pairs():
    config = small_config()
    actions = enumerate_candidate_actions(config)
    assert len(actions) == 6 * 64
    assert all(is_valid_action(action, config) for action in actions)


def test_action_cost_and_invalid_action():
    config = small_config()
    assert action_cost({"action_type": "do_nothing", "region_id": 0}, config) == 0.0
    assert action_cost({"action_type": "inspect_only", "region_id": 0}, config) == 1.0
    assert not is_valid_action({"action_type": "bad_action", "region_id": 0}, config)

