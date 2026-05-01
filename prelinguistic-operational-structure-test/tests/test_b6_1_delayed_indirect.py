from src.b6_hardening.delayed_indirect import backfire_avoidance, credit_assignment_accuracy, delayed_indirect_success, premature_direct_action
from src.b6_hardening.hidden_risk_env import make_hardening_episode


def test_delayed_indirect_success_requires_indirect_action():
    evaluator = {"delay_steps": 3, "backfire_probability": 0.2}
    assert delayed_indirect_success({"action_type": "indirect_stabilize"}, evaluator)
    assert not delayed_indirect_success({"action_type": "apply_local_damping"}, evaluator)


def test_delayed_indirect_credit_assignment_metric_is_finite():
    evaluator = {"delay_steps": 5, "backfire_probability": 0.58}
    value = credit_assignment_accuracy({"action_type": "indirect_stabilize"}, evaluator)
    assert value in {0.0, 1.0}


def test_delayed_indirect_detects_premature_direct_action():
    assert premature_direct_action({"action_type": "apply_local_damping"}, {"delay_steps": 2})
    assert backfire_avoidance(None, {"backfire_probability": 0.6}) == 1.0


def test_delayed_indirect_episode_has_delay_and_indirect_target():
    episode = make_hardening_episode("delayed_indirect", 2, {"b6_1": {"grid_size": 8}}, delay_steps=3)
    assert episode["evaluator_ground_truth"]["delay_steps"] == 3
    assert episode["evaluator_ground_truth"]["oracle_action"]["action_type"] == "indirect_stabilize"

