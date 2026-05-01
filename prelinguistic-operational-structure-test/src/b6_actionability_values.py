from __future__ import annotations

from typing import Any

from .b6_actionability_mask import actionability_violation_type, b6_costs


def compute_actionability_penalty(mask: dict[int, dict[str, Any]], region_id: int, action_type: str, config: dict[str, Any]) -> dict[str, Any]:
    costs = b6_costs(config)
    info = mask[int(region_id)]
    violation = actionability_violation_type(mask, region_id, action_type)
    penalty = 0.0
    if violation in {"non_inspectable", "non_intervenable", "blocked_action"}:
        penalty += costs["wrong_actionability_penalty"]
    if violation == "unsafe" or info.get("unsafe", False):
        penalty += costs["unsafe_penalty"]
    if violation == "irreversible" or info.get("irreversible", False):
        penalty += costs["irreversible_penalty"]
    if violation == "excessive_cost":
        penalty += float(info.get("intervention_cost", 0.0))
    return {"violation_type": violation, "penalty": float(penalty)}


def compute_risk_adjusted_value(epistemic_gain: float, pragmatic_gain: float, actionability_penalty: dict[str, Any], config: dict[str, Any]) -> float:
    return float(epistemic_gain) + float(pragmatic_gain) - float(actionability_penalty.get("penalty", 0.0))


def compute_wrong_actionability_penalty(episode_bundle: dict[str, Any], predicted_action: dict[str, Any], config: dict[str, Any]) -> float:
    mask = episode_bundle["model_input"]["actionability_mask"]
    penalty = compute_actionability_penalty(mask, int(predicted_action["region_id"]), str(predicted_action["action_type"]), config)
    return float(penalty["penalty"])


def compute_oracle_risk_constrained_plan(episode_bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return dict(episode_bundle["oracle_baseline_view"]["oracle_risk_constrained_plan"])
