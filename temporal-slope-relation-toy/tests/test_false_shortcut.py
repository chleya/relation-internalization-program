from src.env_temporal import assign_false_shortcuts, default_delay_map, evaluate_temporal_chain, generate_temporal_sequence
import numpy as np


def test_shortcut_can_be_inverted_without_changing_true_chain():
    seq = generate_temporal_sequence(0, length=8, delay_map=default_delay_map(1), shortcut_correlation=0.9)
    risks_before = [step.risk for step in seq]
    actions_before = [step.optimal_action for step in seq]
    assign_false_shortcuts(seq, np.random.default_rng(1), correlation=0.0)
    evaluate_temporal_chain(seq, delay_map=default_delay_map(1))
    assert [step.risk for step in seq] == risks_before
    assert [step.optimal_action for step in seq] == actions_before
