from __future__ import annotations

from src.partial_metrics import r3_multi_field_cases, run_sequential_case, sequential_update_accuracy


def test_sequential_inspection_updates_one_field_then_acts() -> None:
    true_state, observed, _ = r3_multi_field_cases()[0]
    config = {"inspection_noise": 0.0, "max_inspections_per_case": 2}
    result = run_sequential_case("active_inspection_agent", true_state, observed, config, seed=0)
    assert result["inspections"] == 1
    assert result["final_correct"]
    assert not result["unsafe"]


def test_sequential_update_accuracy_passes_for_active_agent() -> None:
    config = {"inspection_noise": 0.05, "max_inspections_per_case": 2}
    assert sequential_update_accuracy("active_inspection_agent", 0, config) >= 0.75
