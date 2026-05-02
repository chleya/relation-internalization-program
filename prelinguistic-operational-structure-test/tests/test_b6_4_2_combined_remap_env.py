from src.b6_4_2_combined_remap_refinement.combined_remap_env import CONDITIONS, make_b642_episode


def test_b642_env_generates_all_conditions():
    config = {"b6_4_2": {"total_regions": 64}}
    for idx, condition in enumerate(CONDITIONS):
        episode = make_b642_episode(config, idx, condition)
        assert episode["metadata"]["condition"] == condition
        assert "model_input" in episode
        assert episode["evaluator_ground_truth"]["expected_action"]["region_id"] is not None


def test_b642_hidden_mask_removes_answer_like_fields():
    episode = make_b642_episode({"b6_4_2": {}}, 0, "pair_mask_indirect")
    mask = episode["model_input"]["actionability_mask"]
    assert mask
    for info in mask.values():
        assert "indirect_target_region" not in info
        assert "allowed_actions" not in info
        assert "unsafe" not in info
