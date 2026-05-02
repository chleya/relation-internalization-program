from src.b6_2_hardening.env import make_b62_episode
from src.b6_2_hardening.policy import b62_policy


def test_hard_hidden_mask_removes_direct_answer_fields():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 0, "hide_indirect_target", mask_visibility="hard_hidden", hide_indirect_target=True)
    target = episode["evaluator_ground_truth"]["target_region"]
    info = episode["model_input"]["actionability_mask"][target]
    assert "unsafe" not in info
    assert "irreversible" not in info
    assert "indirect_target_region" not in info


def test_hide_indirect_target_does_not_crash_and_records_failure_reason_if_needed():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 0, "hide_indirect_target", mask_visibility="hard_hidden", hide_indirect_target=True)
    output = b62_policy(episode, {})
    assert output["policy_name"] == "b62_policy"
    assert "failure_reason" in output

