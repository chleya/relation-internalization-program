from __future__ import annotations

from src.b21_trace_metrics import (
    false_trace_rejection,
    multi_source_conflict_resolution,
    noisy_trace_robustness,
    trace_compression_survival,
    trace_deletion_specificity_ratio,
    trace_length_extrapolation,
    trace_swap_sensitivity,
)


def test_false_trace_rejection_range():
    assert false_trace_rejection(1, 1, 2) == 1.0
    assert false_trace_rejection(2, 1, 2) == 0.0


def test_trace_swap_sensitivity_range():
    assert trace_swap_sensitivity(1, 2, 2) == 1.0
    assert trace_swap_sensitivity(1, 1, 2) == 0.0


def test_trace_deletion_specificity_ratio_nonnegative():
    assert trace_deletion_specificity_ratio(0.3, 0.1) >= 0.0


def test_multi_source_conflict_resolution_range():
    assert multi_source_conflict_resolution(3, 3, 4) == 1.0
    assert multi_source_conflict_resolution(4, 3, 4) == 0.0


def test_aggregate_metrics_are_probabilities():
    assert 0.0 <= noisy_trace_robustness([1.0, 0.0]) <= 1.0
    assert 0.0 <= trace_length_extrapolation([1.0, 0.0]) <= 1.0
    assert 0.0 <= trace_compression_survival([1.0, 0.0]) <= 1.0
