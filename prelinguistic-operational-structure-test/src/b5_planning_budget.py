from __future__ import annotations

from typing import Any


def make_budget_record(candidate_inspections: int, candidate_interventions: int, rollout_evaluations: int) -> dict[str, int]:
    return {
        "candidate_inspections_evaluated": int(candidate_inspections),
        "candidate_interventions_evaluated": int(candidate_interventions),
        "rollout_evaluations": int(rollout_evaluations),
    }


def planning_budget_compliant(budget_record: dict[str, Any], config: dict[str, Any]) -> bool:
    budget = config.get("b5", {}).get("planning_budget", {})
    return (
        int(budget_record.get("candidate_inspections_evaluated", 0)) <= int(budget.get("max_candidate_inspections", 8))
        and int(budget_record.get("candidate_interventions_evaluated", 0)) <= int(budget.get("max_candidate_interventions", 8))
        and int(budget_record.get("rollout_evaluations", 0)) <= int(budget.get("max_rollout_evaluations", 16))
    )
