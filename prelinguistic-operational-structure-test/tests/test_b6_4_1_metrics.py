from src.b6_4_1_transfer_hardening.hard_remap_env import make_b641_episode
from src.b6_4_1_transfer_hardening.hard_remap_metrics import score_output, summarize_policy


def test_b641_wrong_action_not_full_score():
    episode = make_b641_episode({}, 0, "visual_remap_hard")
    wrong = {"action": {"action_type": "apply_local_damping", "region_id": episode["evaluator_ground_truth"]["decoy_region"]}}
    assert score_output(episode, wrong)["hard_transfer_score"] < 1.0


def test_b641_empty_summary_marks_invalid():
    row = summarize_policy(
        [],
        {"condition": "visual_remap_hard", "seed": 0, "policy_name": "b64_1_transfer_policy"},
        {"oracle": 1.0},
        {"forbidden_reference_count": 0, "poisoned_evaluator_invariance_pass": True, "policy_uses_model_input_only": True},
        clean_score=1.0,
    )
    assert row["hard_transfer_score"] == 0.0
    assert row["no_sample_metric_count"] == 1
    assert row["invalid_metric_count"] == 1

