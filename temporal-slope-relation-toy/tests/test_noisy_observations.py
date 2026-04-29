import numpy as np

from src.env_temporal import TemporalStep, evaluate_temporal_chain
from src.env_uncertainty import apply_observation_uncertainty, finalize_observed_step


def test_noisy_displacement_conflict_marks_takeover():
    seq = [
        TemporalStep(0, "low", "poor", "none", "yes", "dense", "normal", "clear", "safe"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    observed = apply_observation_uncertainty(seq, np.random.default_rng(1), 0.0, 0.0, 0.0)
    observed[0].displacement = "low"
    observed[0].noisy_fields = ("displacement",)
    observed[0].conflicting_fields = ("displacement",)
    finalize_observed_step(observed[0])
    assert observed[0].true_optimal_action == "anchor"
    assert observed[0].observed_automatic_action == "monitor"
    assert observed[0].takeover_required
    assert "PorePressure -> Displacement" in observed[0].uncertain_links

