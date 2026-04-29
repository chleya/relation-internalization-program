from __future__ import annotations

from src.run_r2_1_hardening import run


def test_inspection_budget_penalizes_blanket_inspection() -> None:
    rows = run("configs/r2_1_hardening.yaml")
    by_agent = {row["agent"]: row for row in rows}
    relation_agent = by_agent["relation_specific_uncertainty_agent"]
    blanket = by_agent["missing_always_inspect"]

    assert relation_agent["inspection_precision"] > blanket["inspection_precision"]
    assert relation_agent["unnecessary_inspection_rate"] < blanket["unnecessary_inspection_rate"]
    assert relation_agent["cost_adjusted_success"] >= blanket["cost_adjusted_success"] + 0.10
    assert relation_agent["r2_1_gated_score"] > 0.0
    assert blanket["r2_1_gated_score"] == 0.0
