from src.env_temporal import TemporalStep, evaluate_temporal_chain
from src.env_uncertainty import apply_observation_uncertainty


def test_missing_pore_pressure_can_require_takeover():
    seq = [
        TemporalStep(0, "high", "poor", "none", "no", "dense", "normal", "clear", "safe"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    observed = apply_observation_uncertainty(
        seq,
        rng=__import__("numpy").random.default_rng(0),
        missing_rate=0.0,
        noise_rate=0.0,
        delayed_report_rate=0.0,
    )
    observed[1].pore_pressure = "unknown"
    observed[1].missing_fields = ("pore_pressure",)
    from src.env_uncertainty import finalize_observed_step

    finalize_observed_step(observed[1])
    assert observed[1].takeover_required
    assert observed[1].optimal_action == "takeover"
    assert "Rainfall -> PorePressure" in observed[1].uncertain_links

