import copy

from src.b6_2_hardening.audit import audit_for_episode, policy_source_audit
from src.b6_2_hardening.env import make_b62_episode
from src.b6_2_hardening.policy import b62_policy


def test_policy_source_has_no_forbidden_references():
    audit = policy_source_audit()
    assert audit["forbidden_reference_count"] == 0
    assert audit["policy_uses_model_input_only"] is True


def test_poisoned_evaluator_does_not_change_policy_output():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 0, "missing_mask")
    base = b62_policy(episode, {})
    poisoned = copy.deepcopy(episode)
    poisoned["metadata"]["condition"] = "wrong"
    poisoned["evaluator_ground_truth"]["condition"] = "wrong"
    poisoned["evaluator_ground_truth"]["expected_action"] = None
    poisoned["oracle_baseline_view"]["expected_action"] = None
    assert b62_policy(poisoned, {}) == base
    assert audit_for_episode(episode)["poisoned_ground_truth_invariance_pass"] == 1

