from src.b6_4_1_transfer_hardening.hard_remap_env import CONDITIONS, make_b641_episode


def test_b641_all_hard_conditions_generate_sanitized_model_input():
    for idx, condition in enumerate(CONDITIONS):
        episode = make_b641_episode({}, idx, condition)
        assert episode["evaluator_ground_truth"]["condition"] == condition
        assert "evaluator_ground_truth" not in episode["model_input"]
        assert "oracle_baseline_view" not in episode["model_input"]


def test_visual_hard_remap_removes_public_state_shortcut():
    episode = make_b641_episode({}, 0, "visual_remap_hard")
    visible = episode["model_input"]["visible_state"]
    assert visible["public_state_available"] is False
    assert visible["state_target_hint"] != episode["evaluator_ground_truth"]["target_region"]
    assert episode["model_input"]["previous_trace_state"]["region"] != episode["evaluator_ground_truth"]["target_region"]


def test_mask_hard_remap_removes_answer_like_mask_fields():
    episode = make_b641_episode({}, 0, "mask_visibility_remap_hard")
    mask = episode["model_input"]["actionability_mask"]
    for info in mask.values():
        assert "indirect_target_region" not in info
        assert "unsafe" not in info
        assert "risk_cost" not in info

