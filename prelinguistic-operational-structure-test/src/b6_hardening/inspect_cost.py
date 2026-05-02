from __future__ import annotations

from typing import Any


def should_inspect(model_input: dict[str, Any], inspect_cost: float) -> bool:
    hint = bool(model_input.get("visible_state", {}).get("inspect_need_hint", False))
    if not hint:
        return False
    return float(inspect_cost) <= 0.2


def inspect_cost_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "efficient_inspect_rate": 0.0,
            "over_inspection_rate": 0.0,
            "inspect_cost_adjusted_score": 0.0,
            "required_inspect_recall": 0.0,
            "unnecessary_inspect_rejection_rate": 0.0,
        }
    required = [row for row in records if int(row.get("requires_inspect", 0))]
    unnecessary = [row for row in records if int(row.get("unnecessary_inspect", 0))]
    over = [row for row in records if int(row.get("over_inspection", 0))]
    return {
        "efficient_inspect_rate": sum(int(row.get("inspected", 0)) for row in required) / max(1, len(required)),
        "over_inspection_rate": len(over) / len(records),
        "inspect_cost_adjusted_score": sum(float(row.get("risk_constrained_score", 0.0)) for row in records) / len(records),
        "required_inspect_recall": sum(int(row.get("inspected", 0)) for row in required) / max(1, len(required)),
        "unnecessary_inspect_rejection_rate": sum(1 - int(row.get("inspected", 0)) for row in unnecessary) / max(1, len(unnecessary)),
    }

