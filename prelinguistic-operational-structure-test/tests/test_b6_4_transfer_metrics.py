from src.b6_4_transfer_generalization.remap_env import make_b64_episode
from src.b6_4_transfer_generalization.transfer_metrics import score_output, summarize_policy


def test_wrong_transfer_action_does_not_receive_full_score():
    episode = make_b64_episode({}, 0, "visual_remap")
    wrong = {"inspect": False, "action": {"action_type": "apply_local_damping", "region_id": episode["evaluator_ground_truth"]["decoy_region"]}}
    scored = score_output(episode, wrong)
    assert scored["transfer_score"] < 1.0


def test_empty_records_mark_invalid_not_full_score():
    row = summarize_policy(
        [],
        {"condition": "visual_remap", "seed": 0, "policy_name": "b64_transfer_policy"},
        {"oracle": 1.0},
        {"forbidden_reference_count": 0, "poisoned_evaluator_invariance_pass": True, "policy_uses_model_input_only": True},
        clean_reference_score=1.0,
    )
    assert row["transfer_score"] == 0.0
    assert row["no_sample_metric_count"] == 1
    assert row["invalid_metric_count"] == 1

