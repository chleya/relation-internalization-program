import pytest

from src.b6_risk_constrained_policy import risk_constrained_closed_loop_policy
from src.b6_risk_env import make_b6_risk_constrained_episode
from src.models import make_model


def test_policy_returns_risk_decisions_and_provenance():
    model = make_model("recurrent_flow_checkpoint_model")
    bundle = make_b6_risk_constrained_episode({}, 0, "safe_direct_intervention")
    output = risk_constrained_closed_loop_policy(model, bundle, {})
    assert "inspect_decision" in output
    assert "intervention_decision" in output
    assert "abstain_decision" in output
    assert output["provenance"]["actionability_mask_used"] is True
    assert output["provenance"]["oracle_value_used"] is False


def test_policy_rejects_evaluator_view_shape():
    model = make_model("recurrent_flow_checkpoint_model")
    bundle = make_b6_risk_constrained_episode({}, 0, "safe_direct_intervention")
    with pytest.raises(KeyError):
        risk_constrained_closed_loop_policy(model, {"model_input": bundle["evaluator_ground_truth"]}, {})
