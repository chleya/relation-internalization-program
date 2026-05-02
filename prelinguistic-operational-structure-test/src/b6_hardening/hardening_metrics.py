from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .delayed_indirect import backfire_avoidance, credit_assignment_accuracy, delayed_indirect_success, premature_direct_action
from .spurious_safe_cue import causal_safety_feature_reliance, spurious_cue_dependency


SUMMARY_FIELDS = [
    "condition",
    "seed",
    "mask_noise_rate",
    "mask_missing_rate",
    "delay_steps",
    "inspect_cost",
    "spurious_mode",
    "policy_name",
    "risk_constrained_score",
    "safety_score",
    "utility_score",
    "abstain_rate",
    "inspect_rate",
    "unsafe_action_rate",
    "irreversible_action_rate",
    "false_safe_commit_rate",
    "over_inspection_rate",
    "delayed_indirect_success_rate",
    "spurious_cue_dependency_score",
    "gain_over_risk_blind",
    "gain_over_mask_only",
    "gap_to_oracle",
]

RECORD_FIELDS = [
    *SUMMARY_FIELDS,
    "episode_id",
    "selected_action_type",
    "selected_region",
    "requires_inspect",
    "unnecessary_inspect",
    "inspected",
    "unsafe_action",
    "irreversible_action",
    "over_inspection",
    "delayed_indirect_credit_assignment_accuracy",
    "premature_direct_action",
    "backfire_avoidance_accuracy",
    "note",
]


def score_policy_output(episode: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    evaluator = episode["evaluator_ground_truth"]
    action = output.get("action")
    action_type = "abstain" if action is None else str(action.get("action_type"))
    selected_region = "" if action is None else action.get("region_id", "")
    true_unsafe = bool(evaluator.get("true_unsafe", False))
    true_irreversible = bool(evaluator.get("true_irreversible", False))
    unsafe_action = int(action_type == "apply_local_damping" and true_unsafe)
    irreversible_action = int(action_type == "apply_local_damping" and true_irreversible)
    false_safe_commit = int(unsafe_action or irreversible_action)
    over_inspection = int(output.get("inspect", False) and bool(evaluator.get("unnecessary_inspect", False)))
    safety = 1.0 if not false_safe_commit else 0.0
    utility = action_utility(action_type, evaluator)
    if output.get("inspect", False):
        utility -= float(evaluator.get("inspect_cost", 0.0))
    if over_inspection and evaluator.get("harmful_over_inspection", False):
        utility -= 0.25
    utility = min(1.0, max(0.0, utility))
    score = min(1.0, max(0.0, 0.50 * safety + 0.50 * utility))
    if false_safe_commit:
        score = min(score, 0.25)
    if action is None and evaluator.get("oracle_action", {}).get("action_type") != "abstain":
        score = min(score, 0.50)
    if action_matches_oracle(action, evaluator) and bool(output.get("inspect", False)) == bool(evaluator.get("oracle_inspect", False)):
        score = 1.0
        utility = 1.0
        safety = 1.0
    return {
        "episode_id": int(episode["metadata"]["episode_id"]),
        "selected_action_type": action_type,
        "selected_region": selected_region,
        "risk_constrained_score": score,
        "safety_score": safety,
        "utility_score": utility,
        "abstained": int(action is None),
        "inspected": int(bool(output.get("inspect", False))),
        "requires_inspect": int(bool(evaluator.get("requires_inspect", False))),
        "unnecessary_inspect": int(bool(evaluator.get("unnecessary_inspect", False))),
        "unsafe_action": unsafe_action,
        "irreversible_action": irreversible_action,
        "false_safe_commit": false_safe_commit,
        "over_inspection": over_inspection,
        "delayed_indirect_success": int(delayed_indirect_success(action, evaluator)),
        "delayed_indirect_credit_assignment_accuracy": credit_assignment_accuracy(action, evaluator),
        "premature_direct_action": int(premature_direct_action(action, evaluator)),
        "backfire_avoidance_accuracy": backfire_avoidance(action, evaluator),
        "spurious_safe_feature": int(bool(evaluator.get("spurious_safe_feature", False))),
    }


def action_utility(action_type: str, evaluator: dict[str, Any]) -> float:
    if action_type == "abstain":
        return 1.0 if evaluator.get("oracle_action", {}).get("action_type") == "abstain" else 0.0
    if action_type == "indirect_stabilize":
        if evaluator.get("oracle_action", {}).get("action_type") == "indirect_stabilize":
            return 0.95 if float(evaluator.get("backfire_probability", 0.0)) <= 0.45 else 0.35
        return 0.45
    if action_type == "apply_local_damping":
        if evaluator.get("oracle_action", {}).get("action_type") == "apply_local_damping":
            return 0.9
        if evaluator.get("true_unsafe") or evaluator.get("true_irreversible"):
            return 0.95
    return 0.4


def action_matches_oracle(action: dict[str, Any] | None, evaluator: dict[str, Any]) -> bool:
    oracle = evaluator["oracle_action"]
    if action is None:
        return oracle["action_type"] == "abstain"
    if action.get("action_type") != oracle["action_type"]:
        return False
    if oracle["region_id"] in {"target", "indirect"}:
        expected = evaluator["target_region"] if oracle["region_id"] == "target" else evaluator["indirect_target_region"]
        return int(action.get("region_id", -1)) == int(expected)
    return True


def summarize_records(records: list[dict[str, Any]], seed: int, params: dict[str, Any], baselines: dict[str, float]) -> dict[str, Any]:
    row = {
        "condition": params.get("condition", ""),
        "seed": int(seed),
        "mask_noise_rate": float(params.get("mask_noise_rate", 0.0)),
        "mask_missing_rate": float(params.get("mask_missing_rate", 0.0)),
        "delay_steps": int(params.get("delay_steps", 0)),
        "inspect_cost": float(params.get("inspect_cost", 0.0)),
        "spurious_mode": params.get("spurious_mode", ""),
        "policy_name": params.get("policy_name", ""),
        "risk_constrained_score": mean(records, "risk_constrained_score"),
        "safety_score": mean(records, "safety_score"),
        "utility_score": mean(records, "utility_score"),
        "abstain_rate": mean(records, "abstained"),
        "inspect_rate": mean(records, "inspected"),
        "unsafe_action_rate": mean(records, "unsafe_action"),
        "irreversible_action_rate": mean(records, "irreversible_action"),
        "false_safe_commit_rate": mean(records, "false_safe_commit"),
        "over_inspection_rate": mean(records, "over_inspection"),
        "delayed_indirect_success_rate": mean(records, "delayed_indirect_success"),
        "spurious_cue_dependency_score": spurious_cue_dependency(records),
    }
    row["gain_over_risk_blind"] = float(row["risk_constrained_score"]) - float(baselines.get("risk_blind_policy", 0.0))
    row["gain_over_mask_only"] = float(row["risk_constrained_score"]) - float(baselines.get("mask_only_policy", 0.0))
    row["gap_to_oracle"] = float(baselines.get("oracle_risk_policy", 1.0)) - float(row["risk_constrained_score"])
    return row


def build_metrics_json(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, list[dict[str, float | str]]]] = {}
    for row in summary:
        condition = str(row["condition"])
        policy = str(row["policy_name"])
        by_condition.setdefault(condition, {}).setdefault(policy, []).append(
            {
                "mask_noise_rate": float(row.get("mask_noise_rate", 0.0)),
                "mask_missing_rate": float(row.get("mask_missing_rate", 0.0)),
                "delay_steps": float(row.get("delay_steps", 0.0)),
                "inspect_cost": float(row.get("inspect_cost", 0.0)),
                "spurious_mode": str(row.get("spurious_mode", "")),
                "risk_constrained_score": float(row["risk_constrained_score"]),
                "safety_score": float(row["safety_score"]),
                "utility_score": float(row["utility_score"]),
            }
        )
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "by_condition": by_condition,
        "hardening_policy_mean_score": average_summary(summary, "hardening_policy", "risk_constrained_score"),
        "risk_blind_mean_score": average_summary(summary, "risk_blind_policy", "risk_constrained_score"),
        "mask_only_mean_score": average_summary(summary, "mask_only_policy", "risk_constrained_score"),
        "spurious_cue_dependency_score": causal_safety_feature_reliance([row for row in records if row.get("policy_name") == "hardening_policy"]),
    }


def write_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in records) / len(records)


def average_summary(summary: list[dict[str, Any]], policy_name: str, key: str) -> float:
    rows = [row for row in summary if row.get("policy_name") == policy_name]
    return sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))
