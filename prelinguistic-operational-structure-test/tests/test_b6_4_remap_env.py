from src.b6_4_transfer_generalization.remap_configs import CONDITIONS
from src.b6_4_transfer_generalization.remap_env import make_b64_episode


def test_all_b64_remap_conditions_generate_samples():
    for idx, condition in enumerate(CONDITIONS):
        episode = make_b64_episode({}, idx, condition)
        assert episode["model_input"]["visible_state"]["remap_signature"]
        assert episode["evaluator_ground_truth"]["condition"] == condition
        assert "evaluator_ground_truth" not in episode["model_input"]
        assert "oracle_baseline_view" not in episode["model_input"]


def test_indirect_path_remap_hides_public_indirect_target():
    episode = make_b64_episode({}, 0, "indirect_path_remap")
    target = episode["evaluator_ground_truth"]["indirect_target_region"]
    mask = episode["model_input"]["actionability_mask"]
    public_targets = [info.get("indirect_target_region") for info in mask.values()]
    assert target not in public_targets
    candidates = episode["model_input"]["visible_state"]["candidate_indirect_regions"]
    assert all(row["region_id"] != target for row in candidates)
    assert any(row["region_id"] == target for row in episode["model_input"]["visible_state"]["exploration_history"])


def test_combined_remap_hides_state_and_mask_shortcuts():
    episode = make_b64_episode({}, 0, "combined_remap")
    visible = episode["model_input"]["visible_state"]
    assert visible["public_state_available"] is False
    assert visible["candidate_indirect_regions"] == []
    assert all(info.get("indirect_target_region") is None for info in episode["model_input"]["actionability_mask"].values())
