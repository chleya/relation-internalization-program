from src.b5_clean_episode_view import assert_model_input_is_sanitized
from src.b6_risk_env import B6_EPISODE_TYPES, make_b6_risk_constrained_episode


def test_all_b6_episode_types_generate():
    for idx, episode_type in enumerate(B6_EPISODE_TYPES):
        bundle = make_b6_risk_constrained_episode({}, idx, episode_type)
        assert bundle["evaluator_ground_truth"]["b6_episode_type"] == episode_type
        assert "actionability_mask" in bundle["model_input"]
        assert_model_input_is_sanitized(bundle["model_input"], {})


def test_unsafe_and_indirect_episode_fields():
    unsafe = make_b6_risk_constrained_episode({}, 1, "unsafe_direct_intervention")
    indirect = make_b6_risk_constrained_episode({}, 2, "indirect_only_intervention")
    unsafe_target = unsafe["evaluator_ground_truth"]["target_region"]
    indirect_target = indirect["evaluator_ground_truth"]["target_region"]
    assert unsafe["model_input"]["actionability_mask"][unsafe_target]["unsafe"] is True
    assert indirect["model_input"]["actionability_mask"][indirect_target]["indirectly_intervenable"] is True


def test_abstain_required_has_no_safe_target_action():
    bundle = make_b6_risk_constrained_episode({}, 4, "abstain_required")
    target = bundle["evaluator_ground_truth"]["target_region"]
    info = bundle["model_input"]["actionability_mask"][target]
    assert info["unsafe"] is True
    assert info["intervenable"] is False
