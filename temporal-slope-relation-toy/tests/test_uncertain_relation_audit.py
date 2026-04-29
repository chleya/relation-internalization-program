from src.agents_uncertainty import DelayedRelationChainV3Agent
from src.env_temporal import TemporalStep, evaluate_temporal_chain
from src.env_uncertainty import apply_observation_uncertainty, finalize_observed_step


def test_uncertain_relation_audit_names_field_link_and_takeover():
    seq = [
        TemporalStep(0, "high", "poor", "none", "no", "dense", "normal", "clear", "safe"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    observed = apply_observation_uncertainty(
        seq,
        rng=__import__("numpy").random.default_rng(3),
        missing_rate=0.0,
        noise_rate=0.0,
        delayed_report_rate=0.0,
    )
    observed[1].pore_pressure = "unknown"
    observed[1].missing_fields = ("pore_pressure",)
    finalize_observed_step(observed[1])
    audit = "\n".join(DelayedRelationChainV3Agent().audit_uncertainty(observed, 1))
    assert "takeover" in audit.lower()
    assert "pore_pressure" in audit
    assert "Rainfall -> PorePressure" in audit

