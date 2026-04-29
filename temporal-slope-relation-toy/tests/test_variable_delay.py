from src.env_temporal import TemporalStep, default_delay_map, evaluate_temporal_chain


def make_seq():
    return [
        TemporalStep(0, "high", "poor", "none", "no", "dense", "normal", "clear", "safe"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
        TemporalStep(2, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
    ]


def test_evaluate_temporal_chain_respects_rainfall_delay():
    seq1 = make_seq()
    evaluate_temporal_chain(seq1, delay_map=default_delay_map(1))
    assert seq1[1].pore_pressure == "high"

    seq2 = make_seq()
    evaluate_temporal_chain(seq2, delay_map=default_delay_map(2))
    assert seq2[1].pore_pressure == "low"
    assert seq2[2].pore_pressure == "high"
