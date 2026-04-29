from src.agents_uncertainty import DelayedRelationChainV3Agent
from src.env_temporal import TemporalStep, evaluate_temporal_chain
from src.env_uncertainty import apply_observation_uncertainty, finalize_observed_step


def test_delayed_agent_takes_over_on_critical_missing_observation():
    seq = [
        TemporalStep(0, "high", "poor", "none", "no", "dense", "normal", "clear", "safe"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    observed = apply_observation_uncertainty(
        seq,
        rng=__import__("numpy").random.default_rng(2),
        missing_rate=0.0,
        noise_rate=0.0,
        delayed_report_rate=0.0,
    )
    observed[1].pore_pressure = "unknown"
    observed[1].missing_fields = ("pore_pressure",)
    finalize_observed_step(observed[1])
    assert DelayedRelationChainV3Agent().act(observed, 1) == "takeover"

