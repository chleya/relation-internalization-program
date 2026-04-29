from __future__ import annotations

from src.run_r3_active_inspection import run


def test_budgeted_active_inspection_passes_gates_and_beats_baselines() -> None:
    rows = run("configs/r3_active_inspection.yaml")
    by_agent = {row["agent"]: row for row in rows}
    active = by_agent["active_inspection_agent"]
    first = by_agent["first_missing_inspect"]
    random = by_agent["random_inspect_field"]

    assert active["budgeted_safe_action_rate"] >= 0.80
    assert active["unsafe_automation_rate"] <= 0.10
    assert active["overinspection_rate"] <= 0.25
    assert active["cost_adjusted_success"] >= first["cost_adjusted_success"] + 0.10
    assert active["cost_adjusted_success"] >= random["cost_adjusted_success"] + 0.15
    assert active["r3_gated_score"] > 0.0
