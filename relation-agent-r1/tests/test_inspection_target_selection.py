from __future__ import annotations

from src.partial_metrics import information_gain_efficiency, inspection_target_accuracy


def test_active_inspection_prioritizes_relation_critical_targets() -> None:
    assert inspection_target_accuracy("active_inspection_agent") >= 0.75
    assert information_gain_efficiency("active_inspection_agent") >= 0.70


def test_first_missing_baseline_has_lower_target_accuracy() -> None:
    assert inspection_target_accuracy("active_inspection_agent") > inspection_target_accuracy("first_missing_inspect")
