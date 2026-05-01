from src.b6_actionability_mask import make_actionability_mask
from src.b6_actionability_values import (
    compute_actionability_penalty,
    compute_oracle_risk_constrained_plan,
    compute_risk_adjusted_value,
    compute_wrong_actionability_penalty,
)
from src.b6_risk_env import make_b6_risk_constrained_episode


def test_risk_adjusted_value_penalizes_unsafe_action():
    mask = make_actionability_mask({"metadata": {"target_region": 3, "indirect_target_region": 9, "episode_type": "unsafe_direct_intervention"}}, {}, 0)
    penalty = compute_actionability_penalty(mask, 3, "apply_local_damping", {})
    assert penalty["penalty"] > 0.0
    assert compute_risk_adjusted_value(0.0, 1.0, penalty, {}) < 1.0


def test_wrong_actionability_penalty_and_oracle_plan():
    bundle = make_b6_risk_constrained_episode({}, 0, "unsafe_direct_intervention")
    target = bundle["evaluator_ground_truth"]["target_region"]
    penalty = compute_wrong_actionability_penalty(bundle, {"action_type": "apply_local_damping", "region_id": target}, {})
    assert penalty > 0.0
    assert compute_oracle_risk_constrained_plan(bundle, {}) == bundle["oracle_baseline_view"]["oracle_risk_constrained_plan"]
