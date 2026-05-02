from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .hard_remap_env import CONDITIONS


def review_b6_4_1_results(summary_path: str = "results/b6_4_1_transfer_hardening_summary.csv") -> dict[str, Any]:
    summary = read_csv(Path(summary_path))
    by_condition = {
        condition: {row["policy_name"]: row for row in summary if row["condition"] == condition}
        for condition in sorted({row["condition"] for row in summary})
    }
    missing = [condition for condition in CONDITIONS if condition not in by_condition]
    review_rows = {condition: review_condition(condition, rows) for condition, rows in by_condition.items()}
    shortcut_equivalent = [condition for condition, row in review_rows.items() if condition != "clean_reference" and row["shortcut_equivalent"]]
    review = {
        "submit_ready_as_diagnostic": not missing and integrity(summary)["forbidden_reference_count_max"] == 0,
        "missing_conditions": missing,
        "condition_review": review_rows,
        "shortcut_equivalent_hard_remaps": shortcut_equivalent,
        "hard_remaps_with_baseline_drop": [
            condition for condition, row in review_rows.items() if condition != "clean_reference" and not row["shortcut_equivalent"]
        ],
        "integrity": integrity(summary),
        "remaining_blockers": remaining_blockers(shortcut_equivalent),
    }
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results/b6_4_1_result_review.json").write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_4_1_RESULT_REVIEW.md").write_text(build_review_report(review), encoding="utf-8")
    return review


def review_condition(condition: str, rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    policy = rows.get("b64_1_transfer_policy", {})
    scores = {name: as_float(row.get("hard_transfer_score")) for name, row in rows.items()}
    shortcut = max(scores.get("state_only", 0.0), scores.get("mask_only", 0.0), scores.get("trace_only", 0.0))
    return {
        "b64_1_score": scores.get("b64_1_transfer_policy", 0.0),
        "state_only": scores.get("state_only", 0.0),
        "mask_only": scores.get("mask_only", 0.0),
        "trace_only": scores.get("trace_only", 0.0),
        "oracle": scores.get("oracle", 0.0),
        "hard_baseline_transfer_gap": as_float(policy.get("hard_baseline_transfer_gap")),
        "hard_oracle_gap": as_float(policy.get("hard_oracle_gap")),
        "shortcut_equivalent": scores.get("b64_1_transfer_policy", 0.0) <= shortcut + 0.05,
        "remap_failure_reason": failure_reason(condition, scores.get("b64_1_transfer_policy", 0.0), shortcut),
    }


def failure_reason(condition: str, policy_score: float, shortcut_score: float) -> str:
    if condition == "clean_reference":
        return "reference_not_hard_remap"
    if policy_score <= shortcut_score + 0.05:
        return "shortcut_baseline_matches_policy"
    if policy_score < 0.70:
        return "policy_transfer_drop"
    return "baseline_separation_present"


def integrity(summary: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "forbidden_reference_count_max": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "no_sample_metric_count_total": sum(int(row["no_sample_metric_count"]) for row in summary),
        "remap_leakage_count_total": sum(int(row["remap_leakage_count"]) for row in summary),
        "shortcut_leakage_count_total": sum(int(row["shortcut_leakage_count"]) for row in summary),
        "poisoned_evaluator_invariance_all_pass": all(str(row["poisoned_evaluator_invariance_pass"]).lower() in {"true", "1"} for row in summary),
    }


def remaining_blockers(shortcut_equivalent: list[str]) -> list[str]:
    blockers = []
    if shortcut_equivalent:
        blockers.append(f"shortcut-equivalent hard remaps remain: {shortcut_equivalent}")
    return blockers


def build_review_report(review: dict[str, Any]) -> str:
    lines = [
        "# B6.4.1 Result Review",
        "",
        "## Decision",
        f"- submit_ready_as_diagnostic: {str(review['submit_ready_as_diagnostic']).lower()}",
        "",
        "## Hard Remap Review",
    ]
    for condition, row in review["condition_review"].items():
        lines.append(
            f"- {condition}: b64_1={row['b64_1_score']:.3f}, state={row['state_only']:.3f}, "
            f"mask={row['mask_only']:.3f}, trace={row['trace_only']:.3f}, oracle={row['oracle']:.3f}, "
            f"gap={row['hard_baseline_transfer_gap']:.3f}, shortcut_equivalent={str(row['shortcut_equivalent']).lower()}, "
            f"reason={row['remap_failure_reason']}"
        )
    lines.extend(["", "## Integrity"])
    for key, value in review["integrity"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Remaining Blockers"])
    for blocker in review["remaining_blockers"] or ["No blocking hard-remap issue found for diagnostic branch."]:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "B6.4.1 supports only toy-to-toy hard-remap diagnostic evidence. It does not support real-world generalization, robotics, construction-site autonomy, safety certification, or deployable control.",
            "",
        ]
    )
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

