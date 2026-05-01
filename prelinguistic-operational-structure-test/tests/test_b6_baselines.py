from src.b6_risk_baselines import (
    always_abstain_baseline,
    always_act_baseline,
    oracle_risk_constrained_baseline,
    random_risk_baseline,
    risk_blind_trace_baseline,
    saliency_risk_baseline,
    short_horizon_risk_baseline,
)
from src.b6_risk_env import make_b6_risk_constrained_episode
from src.models import make_model


def test_b6_baselines_run():
    model = make_model("recurrent_flow_checkpoint_model")
    bundle = make_b6_risk_constrained_episode({}, 0, "risk_blind_trap")
    assert random_risk_baseline(bundle, {}, 0)
    assert saliency_risk_baseline(bundle, {})
    assert short_horizon_risk_baseline(bundle, {})
    assert risk_blind_trace_baseline(model, bundle, {})
    assert always_act_baseline(bundle, {})
    assert always_abstain_baseline(bundle, {})
    assert oracle_risk_constrained_baseline(bundle, {})
