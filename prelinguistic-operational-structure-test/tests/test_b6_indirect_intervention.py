from src.b6_indirect_intervention import apply_indirect_intervention, evaluate_indirect_intervention_choice, find_indirect_intervention_candidates
from src.b6_risk_constrained_policy import risk_constrained_closed_loop_policy
from src.b6_risk_env import make_b6_risk_constrained_episode
from src.models import make_model


def test_indirect_candidates_found_for_indirect_episode():
    bundle = make_b6_risk_constrained_episode({}, 2, "indirect_only_intervention")
    target = bundle["evaluator_ground_truth"]["target_region"]
    candidates = find_indirect_intervention_candidates(bundle, target, {})
    assert candidates


def test_indirect_evaluation_and_apply():
    model = make_model("field_memory_model")
    bundle = make_b6_risk_constrained_episode({}, 2, "indirect_only_intervention")
    output = risk_constrained_closed_loop_policy(model, bundle, {})
    metrics = evaluate_indirect_intervention_choice(output, bundle["evaluator_ground_truth"], {})
    applied = apply_indirect_intervention(bundle, output["intervention_decision"]["action"], {})
    assert metrics["indirect_intervention_accuracy"] >= 0.0
    assert applied["indirect_value"] >= 0.0
