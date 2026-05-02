from src.b6_2_hardening.delayed_credit_assignment import (
    assign_credit_to_indirect_action,
    register_pending_indirect_action,
)
from src.b6_2_hardening.env import make_b62_episode
from src.b6_2_hardening.metrics import score_output
from src.b6_2_hardening.policy import b62_policy


def test_successful_delay5_indirect_action_is_credited():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 2003, "delayed_indirect", delay_steps=5)
    output = b62_policy(episode, {})
    assert output["action"]["action_type"] == "indirect_stabilize"
    assert output["delayed_credit"]["credit_assigned"] is True
    assert score_output(episode, output)["delayed_credit_success"] == 1


def test_no_effect_and_backfire_are_not_success():
    pending = register_pending_indirect_action(
        {"action_type": "indirect_stabilize", "region_id": 5},
        {"visible_state": {"delay_steps": 5, "expected_delayed_effect_signature": "stabilize_target"}},
        {},
    )
    no_effect = assign_credit_to_indirect_action(
        pending,
        [{"after_steps": 5, "effect_signature": "none", "no_effect": True, "backfire": False}],
        {},
    )
    backfire = assign_credit_to_indirect_action(
        pending,
        [{"after_steps": 5, "effect_signature": "destabilize_target", "no_effect": False, "backfire": True}],
        {},
    )
    assert no_effect["credit_assigned"] is False
    assert backfire["credit_assigned"] is False


def test_abstain_does_not_count_as_indirect_success():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 2007, "delayed_indirect", delay_steps=5)
    output = b62_policy(episode, {})
    assert output["action"] is None
    assert score_output(episode, output)["delayed_credit_success"] == 0
