from src.b6_4_transfer_generalization.remap_env import make_b64_episode
from src.b6_4_transfer_generalization.transfer_policy import audit_policy_integrity, b64_transfer_policy


def test_b64_policy_source_has_no_forbidden_references():
    episode = make_b64_episode({}, 0, "combined_remap")
    audit = audit_policy_integrity(episode, {})
    assert audit["forbidden_reference_count"] == 0
    assert audit["poisoned_evaluator_invariance_pass"] is True
    assert audit["policy_uses_model_input_only"] is True


def test_poisoned_evaluator_does_not_change_b64_output():
    episode = make_b64_episode({}, 0, "indirect_path_remap")
    original = b64_transfer_policy(episode, {})
    poisoned = {
        **episode,
        "evaluator_ground_truth": {**episode["evaluator_ground_truth"], "expected_action": None, "condition": "poisoned"},
        "metadata": {**episode["metadata"], "condition": "poisoned"},
    }
    changed = b64_transfer_policy(poisoned, {})
    assert comparable(original) == comparable(changed)


def comparable(output):
    action = output.get("action") or {}
    return action.get("action_type"), action.get("region_id"), bool(output.get("inspect")), output.get("transfer_source")

