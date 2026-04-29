from src.env_temporal import evaluate_temporal_chain, generate_sequence, TemporalStep


def test_rainfall_delays_pore_pressure():
    seq = [
        TemporalStep(0, "high", "poor", "none", "no", "dense", "normal", "clear", "safe"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    assert seq[0].pore_pressure == "low"
    assert seq[1].pore_pressure == "high"
    assert seq[1].optimal_action == "drain"


def test_generate_sequence_has_surface_labels():
    seq = generate_sequence(0, length=6, mode="base")
    assert len(seq) == 6
    assert seq[0].weather_label in {"clear", "storm"}
