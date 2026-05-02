from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .combined_remap_env import ABLATION_CONDITIONS, CONDITIONS, PAIRWISE_CONDITIONS, TRIPLE_CONDITIONS


def review_b6_4_2_results(summary_path: str = "results/b6_4_2_combined_refinement_summary.csv") -> dict[str, Any]:
    summary = read_csv(Path(summary_path))
    by_condition = {
        condition: {row["policy_name"]: row for row in summary if row["condition"] == condition}
        for condition in sorted({row["condition"] for row in summary})
    }
    missing = [condition for condition in CONDITIONS if condition not in by_condition]
    condition_review = {condition: review_condition(condition, rows) for condition, rows in by_condition.items()}
    integrity = integrity_review(summary)
    combined = condition_review.get("combined_remap_hard_reference", {})
    failure_source = infer_failure_source(condition_review)
    gate = "STAY_IN_B6_REFINEMENT" if float(combined.get("oracle_gap", 1.0)) > 0.15 else "PROCEED_TO_B7_ALLOWED"
    review = {
        "submit_ready_as_diagnostic": not missing and integrity["forbidden_reference_count_max"] == 0 and integrity["invalid_metric_count_total"] == 0,
        "missing_conditions": missing,
        "condition_review": condition_review,
        "pairwise_results": {condition: condition_review.get(condition, {}) for condition in PAIRWISE_CONDITIONS},
        "triple_results": {condition: condition_review.get(condition, {}) for condition in TRIPLE_CONDITIONS},
        "mechanism_ablation_results": {condition: condition_review.get(condition, {}) for condition in ABLATION_CONDITIONS},
        "combined_failure_source": failure_source,
        "combined_oracle_gap": float(combined.get("oracle_gap", 0.0)),
        "integrity": integrity,
        "gate_judgment": gate,
        "gate_reason": "combined_remap_hard remains a B6.x refinement blocker" if gate == "STAY_IN_B6_REFINEMENT" else "combined remap gap is low enough for gate review",
        "remaining_blockers": remaining_blockers(condition_review, integrity),
    }
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results/b6_4_2_result_review.json").write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_4_2_RESULT_REVIEW.md").write_text(build_review_report(review), encoding="utf-8")
    return review


def review_condition(condition: str, rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    score = lambda policy: as_float(rows.get(policy, {}).get("combined_refinement_score"))
    policy = score("b64_2_combined_policy")
    state = score("state_only")
    mask = score("mask_only")
    trace = score("trace_only")
    oracle = score("oracle")
    shortcut = max(state, mask, trace)
    return {
        "policy_score": policy,
        "b64_1_reference": score("b64_1_policy_reference"),
        "state_only": state,
        "mask_only": mask,
        "trace_only": trace,
        "random": score("random"),
        "always_abstain": score("always_abstain"),
        "conservative": score("conservative_uncertainty"),
        "oracle": oracle,
        "baseline_gap": policy - shortcut,
        "oracle_gap": oracle - policy,
        "shortcut_explainable": policy <= shortcut + 0.05,
        "failure_reason": failure_reason(condition, policy, shortcut, oracle),
    }


def failure_reason(condition: str, policy: float, shortcut: float, oracle: float) -> str:
    if condition in ABLATION_CONDITIONS and oracle - policy > 0.20:
        return "mechanism_removed_policy_collapses"
    if policy <= shortcut + 0.05:
        return "shortcut_baseline_matches_policy"
    if oracle - policy > 0.25:
        return "large_oracle_gap"
    return "baseline_separation_present"


def infer_failure_source(condition_review: dict[str, dict[str, Any]]) -> str:
    candidates = {
        "dynamics_delay_credit_interaction": min(
            condition_review.get("pair_dynamics_delay", {}).get("policy_score", 0.0),
            condition_review.get("triple_dynamics_delay_indirect", {}).get("policy_score", 0.0),
        ),
        "mask_indirect_candidate_search_interaction": min(
            condition_review.get("pair_mask_indirect", {}).get("policy_score", 0.0),
            condition_review.get("triple_visual_mask_indirect", {}).get("policy_score", 0.0),
        ),
        "visual_risk_interaction": condition_review.get("pair_visual_risk", {}).get("policy_score", 0.0),
    }
    return min(candidates.items(), key=lambda item: item[1])[0]


def integrity_review(summary: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "forbidden_reference_count_max": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "no_sample_metric_count_total": sum(int(row["no_sample_metric_count"]) for row in summary),
        "combined_leakage_count_total": sum(int(row["combined_leakage_count"]) for row in summary),
        "shortcut_leakage_count_total": sum(int(row["shortcut_leakage_count"]) for row in summary),
        "poisoned_evaluator_invariance_all_pass": all(str(row["poisoned_evaluator_invariance_pass"]).lower() in {"true", "1"} for row in summary),
    }


def remaining_blockers(condition_review: dict[str, dict[str, Any]], integrity: dict[str, Any]) -> list[str]:
    blockers = []
    combined_gap = condition_review.get("combined_remap_hard_reference", {}).get("oracle_gap", 0.0)
    if combined_gap > 0.15:
        blockers.append(f"combined_remap_hard oracle gap remains {combined_gap:.3f}")
    shortcuts = [
        condition
        for condition, row in condition_review.items()
        if row.get("shortcut_explainable") and condition not in ABLATION_CONDITIONS
    ]
    if shortcuts:
        blockers.append(f"shortcut-explainable combined remaps remain: {shortcuts}")
    if integrity["forbidden_reference_count_max"]:
        blockers.append("forbidden reference detected")
    return blockers


def build_review_report(review: dict[str, Any]) -> str:
    lines = [
        "# B6.4.2 Result Review",
        "",
        "## Decision",
        f"- submit_ready_as_diagnostic: {str(review['submit_ready_as_diagnostic']).lower()}",
        f"- gate_judgment: {review['gate_judgment']}",
        f"- gate_reason: {review['gate_reason']}",
        "",
        "## Pairwise Results",
    ]
    for condition, row in review["pairwise_results"].items():
        lines.append(format_condition(condition, row))
    lines.extend(["", "## Triple Results"])
    for condition, row in review["triple_results"].items():
        lines.append(format_condition(condition, row))
    lines.extend(["", "## Mechanism Ablations"])
    for condition, row in review["mechanism_ablation_results"].items():
        lines.append(format_condition(condition, row))
    lines.extend(
        [
            "",
            "## Failure Attribution",
            f"- combined_failure_source: {review['combined_failure_source']}",
            f"- combined_oracle_gap: {review['combined_oracle_gap']:.3f}",
            "",
            "## Integrity",
        ]
    )
    for key, value in review["integrity"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Remaining Blockers"])
    for blocker in review["remaining_blockers"] or ["No leakage or metric blocker found; combined gap is diagnostic focus."]:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "B6.4.2 is toy-to-toy combined-remap diagnostic evidence only. It does not support real-world transfer, robotics, safety certification, construction-site autonomy, or deployable control.",
            "",
        ]
    )
    return "\n".join(lines)


def format_condition(condition: str, row: dict[str, Any]) -> str:
    return (
        f"- {condition}: policy={row.get('policy_score', 0.0):.3f}, state={row.get('state_only', 0.0):.3f}, "
        f"mask={row.get('mask_only', 0.0):.3f}, trace={row.get('trace_only', 0.0):.3f}, "
        f"oracle={row.get('oracle', 0.0):.3f}, gap={row.get('baseline_gap', 0.0):.3f}, "
        f"oracle_gap={row.get('oracle_gap', 0.0):.3f}, reason={row.get('failure_reason', 'none')}"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
