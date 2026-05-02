from src.b6_4_1_transfer_hardening.hard_remap_env import make_b641_episode
from src.b6_4_1_transfer_hardening.hard_remap_policy import audit_policy_integrity, b641_policy


def test_b641_policy_has_no_forbidden_references():
    episode = make_b641_episode({}, 0, "combined_remap_hard")
    audit = audit_policy_integrity(episode, {})
    assert audit["forbidden_reference_count"] == 0
    assert audit["poisoned_evaluator_invariance_pass"] is True
    assert audit["policy_uses_model_input_only"] is True


def test_poisoned_evaluator_does_not_change_b641_policy():
    episode = make_b641_episode({}, 0, "risk_cue_remap_hard")
    original = b641_policy(episode, {})
    poisoned = {
        **episode,
        "evaluator_ground_truth": {**episode["evaluator_ground_truth"], "expected_action": None, "condition": "poisoned"},
        "metadata": {**episode["metadata"], "condition": "poisoned"},
    }
    changed = b641_policy(poisoned, {})
    assert comparable(original) == comparable(changed)


def comparable(output):
    action = output.get("action") or {}
    return action.get("action_type"), action.get("region_id"), output.get("transfer_source")
