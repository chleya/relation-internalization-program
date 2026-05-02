from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .remap_configs import CONDITIONS


def review_b6_4_results(summary_path: str = "results/b6_4_transfer_summary.csv") -> dict[str, Any]:
    summary = read_csv(Path(summary_path))
    b64 = [row for row in summary if row["policy_name"] == "b64_transfer_policy"]
    conditions_present = sorted({row["condition"] for row in summary})
    missing = [condition for condition in CONDITIONS if condition not in conditions_present]
    per_condition = {
        condition: {row["policy_name"]: as_float(row["transfer_score"]) for row in summary if row["condition"] == condition}
        for condition in conditions_present
    }
    failing = [row["condition"] for row in b64 if row["condition"] != "clean_reference" and as_float(row["transfer_score"]) < 0.70]
    review = {
        "submit_ready_as_diagnostic": not missing and max((int(row["forbidden_reference_count"]) for row in summary), default=0) == 0,
        "conditions": conditions_present,
        "missing_conditions": missing,
        "per_condition_scores": per_condition,
        "transfer_evidence": {
            "mean_b64_transfer_score": mean(b64, "transfer_score"),
            "mean_baseline_transfer_gap": mean(b64, "baseline_transfer_gap"),
            "mean_oracle_gap": mean(b64, "oracle_gap"),
            "failing_remaps": failing,
            "shortcut_equivalent_remaps": shortcut_equivalent_remaps(per_condition),
        },
        "shortcut_review": shortcut_review(per_condition),
        "integrity": {
            "forbidden_reference_count_max": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
            "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
            "no_sample_metric_count_total": sum(int(row["no_sample_metric_count"]) for row in summary),
            "poisoned_evaluator_invariance_all_pass": all(str(row["poisoned_evaluator_invariance_pass"]).lower() in {"true", "1"} for row in summary),
        },
        "claim_boundary": "toy_to_toy_transfer_only",
    }
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results/b6_4_result_review.json").write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_4_RESULT_REVIEW.md").write_text(build_review_report(review), encoding="utf-8")
    return review


def shortcut_review(per_condition: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition, scores in per_condition.items():
        b64 = scores.get("b64_transfer_policy", 0.0)
        shortcut = max(scores.get("state_only", 0.0), scores.get("mask_only", 0.0), scores.get("trace_only", 0.0))
        out[condition] = {
            "b64_score": b64,
            "shortcut_best": shortcut,
            "b64_beats_shortcuts": b64 > shortcut,
            "oracle_gap": scores.get("oracle", 1.0) - b64,
        }
    return out


def shortcut_equivalent_remaps(per_condition: dict[str, dict[str, float]]) -> list[str]:
    out = []
    for condition, scores in per_condition.items():
        if condition == "clean_reference":
            continue
        b64 = scores.get("b64_transfer_policy", 0.0)
        shortcut = max(scores.get("state_only", 0.0), scores.get("mask_only", 0.0), scores.get("trace_only", 0.0))
        if b64 <= shortcut + 1e-9:
            out.append(condition)
    return sorted(out)


def build_review_report(review: dict[str, Any]) -> str:
    lines = [
        "# B6.4 Result Review",
        "",
        "## Decision",
        f"- submit_ready_as_diagnostic: {str(review['submit_ready_as_diagnostic']).lower()}",
        "",
        "## Transfer Evidence",
    ]
    for key, value in review["transfer_evidence"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Shortcut Review"])
    for condition, row in review["shortcut_review"].items():
        lines.append(f"- {condition}: b64={row['b64_score']:.3f}, shortcut_best={row['shortcut_best']:.3f}, oracle_gap={row['oracle_gap']:.3f}")
    lines.extend(
        [
            "",
            "## Caveats",
            f"- shortcut_equivalent_remaps: {review['transfer_evidence']['shortcut_equivalent_remaps']}",
            "- A high B6.4 score is not sufficient transfer evidence when state_only, mask_only, or trace_only match the transfer policy.",
            "- Remaps with zero oracle gap but no baseline separation should be treated as shortcut-explainable.",
            "",
            "## Interpretation",
            "B6.4 can support only toy-to-toy transfer diagnostics. If remaps fail or shortcuts remain strong, interpret the result as split-specific rather than general operational structure transfer.",
            "",
            "## Claim Boundary",
            "No real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control claim is supported.",
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


def mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(as_float(row.get(key)) for row in rows) / len(rows)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
