from src.b6_actionability_mask import (
    actionability_violation_type,
    get_region_actionability,
    is_action_allowed,
    make_actionability_mask,
    sanitize_actionability_mask_for_model,
)


def test_actionability_mask_contains_required_fields():
    bundle = {"metadata": {"target_region": 3, "indirect_target_region": 9, "episode_type": "safe_direct_intervention"}}
    mask = make_actionability_mask(bundle, {}, 0)
    info = get_region_actionability(mask, 3)
    for key in ["observable", "inspectable", "directly_intervenable", "indirectly_intervenable", "unsafe", "irreversible", "costly", "allowed_actions"]:
        assert key in info


def test_sanitized_mask_removes_evaluator_fields():
    mask = {0: {"region_id": 0, "observable": True, "oracle_best_action": "x"}}
    sanitized = sanitize_actionability_mask_for_model(mask, {})
    assert "oracle_best_action" not in sanitized[0]


def test_action_allowed_and_violation():
    bundle = {"metadata": {"target_region": 3, "indirect_target_region": 9, "episode_type": "unsafe_direct_intervention"}}
    mask = make_actionability_mask(bundle, {}, 0)
    assert not is_action_allowed(mask, 3, "apply_local_damping")
    assert actionability_violation_type(mask, 3, "apply_local_damping") == "unsafe"
