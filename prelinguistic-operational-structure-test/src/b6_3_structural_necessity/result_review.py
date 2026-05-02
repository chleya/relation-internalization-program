from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def run_result_review(
    summary_path: Path = Path("results/b6_3_structural_necessity_summary.csv"),
    json_output_path: Path = Path("results/b6_3_result_review.json"),
    report_output_path: Path = Path("reports/B6_3_RESULT_REVIEW.md"),
) -> dict[str, Any]:
    rows = read_csv(summary_path)
    review = build_review(rows)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    report_output_path.write_text(build_report(review), encoding="utf-8")
    return review


def build_review(rows: list[dict[str, str]]) -> dict[str, Any]:
    conditions = sorted({row["condition"] for row in rows})
    policies = sorted({row["policy_name"] for row in rows})
    b63 = [row for row in rows if row["policy_name"] == "b63_policy"]
    drops = {
        row["ablation_name"]: avg([r for r in rows if r["ablation_name"] == row["ablation_name"]], "observed_performance_drop")
        for row in rows
        if row["ablation_name"] not in {"reference", "baseline", "reference_b62"}
    }
    by_condition_policy = {(row["condition"], row["policy_name"]): row for row in rows}
    review = {
        "conditions": conditions,
        "policies": policies,
        "mean_b63_score": avg(b63, "risk_constrained_score"),
        "mean_structural_necessity_score": avg([row for row in rows if row["policy_name"].startswith("b63_")], "structural_necessity_score"),
        "drop_by_ablation": dict(sorted(drops.items())),
        "mechanism_necessity": {
            "trace": max(drops.get("remove_trace", 0.0), drops.get("shuffle_trace", 0.0), drops.get("corrupt_trace", 0.0)),
            "history": drops.get("remove_history", 0.0),
            "feedback_update": drops.get("freeze_feedback_update", 0.0),
            "delayed_credit_buffer": drops.get("disable_delayed_credit_buffer", 0.0),
            "candidate_search": drops.get("disable_candidate_search", 0.0),
            "risk_cue": drops.get("remove_risk_cue", 0.0),
            "public_state": drops.get("hide_public_state_cue", 0.0),
        },
        "trace_repair": {
            "wrong_trace_b63": score(by_condition_policy, "wrong_trace", "b63_policy"),
            "wrong_trace_state_only": score(by_condition_policy, "wrong_trace", "state_only"),
            "wrong_trace_mask_only": score(by_condition_policy, "wrong_trace", "mask_only"),
            "wrong_trace_state_ambiguous_b63": score(by_condition_policy, "wrong_trace_state_ambiguous", "b63_policy"),
            "hide_public_state_cue_b63": score(by_condition_policy, "hide_public_state_cue", "b63_policy"),
        },
        "integrity": {
            "forbidden_reference_count_max": max((int(float(row["forbidden_reference_count"])) for row in rows), default=0),
            "invalid_metric_count_total": sum(int(float(row["invalid_metric_count"])) for row in rows),
            "poisoned_ground_truth_invariance_all_pass": all(int(float(row["poisoned_ground_truth_invariance_pass"])) == 1 for row in rows),
        },
    }
    review["submit_ready_as_diagnostic"] = (
        review["integrity"]["forbidden_reference_count_max"] == 0
        and review["integrity"]["invalid_metric_count_total"] == 0
        and len(conditions) >= 13
    )
    review["unresolved"] = unresolved(review)
    return review


def build_report(review: dict[str, Any]) -> str:
    lines = [
        "# B6.3 Result Review",
        "",
        "## Decision",
        f"- submit_ready_as_diagnostic: {str(review['submit_ready_as_diagnostic']).lower()}",
        "",
        "## Structural Necessity",
        "| mechanism | drop | interpretation |",
        "| --- | ---: | --- |",
    ]
    for mechanism, drop in review["mechanism_necessity"].items():
        interpretation = "necessity_evidence" if drop >= 0.10 else "not_proven_necessary"
        lines.append(f"| {mechanism} | {drop:.3f} | {interpretation} |")
    lines.extend(
        [
            "",
            "## Trace Repair",
            f"- wrong_trace b63: {review['trace_repair']['wrong_trace_b63']:.3f}",
            f"- wrong_trace state_only: {review['trace_repair']['wrong_trace_state_only']:.3f}",
            f"- wrong_trace mask_only: {review['trace_repair']['wrong_trace_mask_only']:.3f}",
            f"- wrong_trace_state_ambiguous b63: {review['trace_repair']['wrong_trace_state_ambiguous_b63']:.3f}",
            f"- hide_public_state_cue b63: {review['trace_repair']['hide_public_state_cue_b63']:.3f}",
            "",
            "## Integrity",
            f"- forbidden_reference_count_max: {review['integrity']['forbidden_reference_count_max']}",
            f"- invalid_metric_count_total: {review['integrity']['invalid_metric_count_total']}",
            f"- poisoned_ground_truth_invariance_all_pass: {str(review['integrity']['poisoned_ground_truth_invariance_all_pass']).lower()}",
            "",
            "## Unresolved",
        ]
    )
    for item in review["unresolved"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "B6.3 is a toy diagnostic for structural necessity. It does not support real-world risk intelligence, safety certification, robotics, construction-site autonomy, or engineering deployment.",
            "",
        ]
    )
    return "\n".join(lines)


def unresolved(review: dict[str, Any]) -> list[str]:
    items = []
    if review["trace_repair"]["wrong_trace_b63"] <= review["trace_repair"]["wrong_trace_state_only"]:
        items.append("wrong_trace remains explainable by state_only.")
    if review["mechanism_necessity"]["feedback_update"] < 0.10:
        items.append("feedback update necessity is not proven.")
    if review["mechanism_necessity"]["history"] < 0.10:
        items.append("history necessity is not proven.")
    if review["mechanism_necessity"]["candidate_search"] < 0.10:
        items.append("candidate search necessity is not proven.")
    if review["mechanism_necessity"]["delayed_credit_buffer"] < 0.10:
        items.append("delayed credit buffer necessity is not proven.")
    return items


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def avg(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)


def score(rows: dict[tuple[str, str], dict[str, str]], condition: str, policy: str) -> float:
    return float(rows.get((condition, policy), {}).get("risk_constrained_score", 0.0) or 0.0)
