import copy

from src.b6_3_structural_necessity.audit import policy_source_audit
from src.b6_3_structural_necessity.env import make_b63_episode
from src.b6_3_structural_necessity.structural_policy import b63_policy


def test_b63_policy_source_has_no_forbidden_reference():
    audit = policy_source_audit()
    assert audit["forbidden_reference_count"] == 0
    assert audit["policy_uses_model_input_only"] is True


def test_poisoned_evaluator_and_metadata_do_not_change_output():
    episode = make_b63_episode({"b6_3": {"grid_size": 8}}, 0, "wrong_trace")
    base = b63_policy(episode, {})
    poisoned = copy.deepcopy(episode)
    poisoned["evaluator_ground_truth"]["expected_action"] = None
    poisoned["evaluator_ground_truth"]["condition"] = "poisoned"
    poisoned["metadata"]["episode_type"] = "poisoned"
    poisoned["oracle_baseline_view"]["expected_action"] = None
    assert b63_policy(poisoned, {}) == base
