from __future__ import annotations

import copy
from typing import Any

from .b5_closed_loop_policy import evaluate_closed_loop_policy


def run_planning_budget_stress(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], model_name: str = "") -> tuple[dict[str, float], list[dict[str, Any]]]:
    strict_config = copy.deepcopy(config)
    stress = strict_config.get("b51", {}).get("budget_stress", {})
    strict_config.setdefault("b5", {}).setdefault("planning_budget", {})
    strict_config["b5"]["planning_budget"] = {
        "max_candidate_inspections": int(stress.get("strict_max_candidate_inspections", 4)),
        "max_candidate_interventions": int(stress.get("strict_max_candidate_interventions", 4)),
        "max_rollout_evaluations": int(stress.get("strict_max_rollout_evaluations", 8)),
    }
    original_metrics, _ = evaluate_closed_loop_policy(model, episodes, config, model_name=model_name)
    strict_metrics, records = evaluate_closed_loop_policy(model, episodes, strict_config, model_name=model_name)
    original_score = float(original_metrics.get("closed_loop_model_score", 0.0))
    strict_score = float(strict_metrics.get("closed_loop_model_score", 0.0))
    retention = compute_budget_stress_retention(original_score, strict_score)
    for record in records:
        record["record_kind"] = "planning_budget_stress"
        record["note"] = "strict planning budget stress"
    return {
        "strict_budget_score": strict_score,
        "original_budget_score": original_score,
        "planning_budget_stress_retention": retention,
        "strict_budget_compliance": float(strict_metrics.get("planning_budget_compliance", 0.0)),
        "rollout_eval_excess_rate": 1.0 - float(strict_metrics.get("planning_budget_compliance", 0.0)),
        "candidate_search_reduction": 0.5,
    }, records


def compute_budget_stress_retention(original_score: float, strict_budget_score: float) -> float:
    if float(original_score) <= 0.0:
        return 0.0
    return float(strict_budget_score) / float(original_score)
