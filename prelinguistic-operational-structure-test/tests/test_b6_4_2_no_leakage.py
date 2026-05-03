import inspect

from src.b6_4_2_combined_remap_refinement.combined_policy import b642_policy
from src.b6_4_2_combined_remap_refinement.combined_remap_env import make_b642_episode


def test_b642_policy_source_has_no_forbidden_decision_references():
    source = inspect.getsource(b642_policy)
    forbidden = [
        "evaluator_ground_truth",
        "oracle_baseline_view",
        "expected_decision",
        "failure_source",
        "metadata",
    ]
    assert not any(token in source for token in forbidden)


def test_b642_policy_invariant_to_poisoned_evaluator_fields():
    episode = make_b642_episode({"b6_4_2": {}}, 0, "pair_visual_risk")
    original = b642_policy(episode, {})
    poisoned = {
        **episode,
        "evaluator_ground_truth": {**episode["evaluator_ground_truth"], "expected_action": None, "failure_source": "poisoned"},
        "metadata": {**episode["metadata"], "condition": "poisoned"},
    }
    changed = b642_policy(poisoned, {})
    assert original["action"] == changed["action"]
    assert original["transfer_source"] == changed["transfer_source"]


def test_b642_hidden_indirect_target_not_in_model_input():
    episode = make_b642_episode({"b6_4_2": {}}, 0, "pair_mask_indirect")
    text = repr(episode["model_input"])
    assert "indirect_target_region" not in text
    assert "oracle" not in text
