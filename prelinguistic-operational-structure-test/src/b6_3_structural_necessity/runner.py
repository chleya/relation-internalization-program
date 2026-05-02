from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ablations import EXPECTED_AFFECTED_SPLITS
from .audit import audit_for_episode
from .baselines import ABLATION_POLICY_NAMES, BASELINE_NAMES, get_policy
from .env import make_b63_datasets
from .metrics import RECORD_FIELDS, SUMMARY_FIELDS, mean, score_policy_output, summarize_policy


POLICY_NAMES = [*BASELINE_NAMES, *ABLATION_POLICY_NAMES]

POLICY_TO_ABLATION = {
    "b62_policy": "reference_b62",
    "b63_policy": "reference",
    "state_only": "baseline",
    "mask_only": "baseline",
    "trace_only": "baseline",
    "random": "baseline",
    "always_abstain": "baseline",
    "conservative_uncertainty": "baseline",
    "oracle": "baseline",
    "b63_no_trace": "remove_trace",
    "b63_shuffled_trace": "shuffle_trace",
    "b63_corrupt_trace": "corrupt_trace",
    "b63_no_feedback_update": "freeze_feedback_update",
    "b63_no_history": "remove_history",
    "b63_no_risk_cue": "remove_risk_cue",
    "b63_no_candidate_search": "disable_candidate_search",
    "b63_no_credit_buffer": "disable_delayed_credit_buffer",
    "b63_no_inspection_recovery": "disable_inspection_recovery",
}


def run_b6_3_structural_necessity(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    datasets = make_b63_datasets(config, seed)
    summary: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for condition, episodes in datasets.items():
        if not episodes:
            summary.append(empty_summary(condition, seed))
            continue
        audit = audit_for_episode(episodes[0])
        rows_by_policy: dict[str, list[dict[str, Any]]] = {}
        for policy_name in POLICY_NAMES:
            policy = get_policy(policy_name)
            policy_rows = []
            for episode in episodes:
                output = policy(episode, config)
                scored = score_policy_output(episode, output)
                record = {
                    **base_context(episode, seed, policy_name),
                    **scored,
                    "ablation_name": POLICY_TO_ABLATION[policy_name],
                    "affected_mechanism": mechanism_for_policy(policy_name),
                    "interpretation": interpretation_for(policy_name, condition),
                }
                policy_rows.append(record)
                records.append(record)
            rows_by_policy[policy_name] = policy_rows

        baseline_scores = {policy: mean(rows, "risk_constrained_score") for policy, rows in rows_by_policy.items()}
        reference_score = baseline_scores.get("b63_policy", 0.0)
        for policy_name, rows in rows_by_policy.items():
            context = base_context(episodes[0], seed, policy_name)
            summary.append(
                summarize_policy(
                    rows,
                    context,
                    baseline_scores,
                    reference_score,
                    POLICY_TO_ABLATION[policy_name],
                    audit,
                )
            )
    metrics = build_metrics(summary, records)
    return summary, records, metrics


def write_b6_3_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b6_3_structural_necessity_summary.csv"), summary, SUMMARY_FIELDS)
    write_csv(Path("results/b6_3_structural_necessity_records.csv"), records, RECORD_FIELDS)
    Path("results/b6_3_structural_necessity_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_3_STRUCTURAL_NECESSITY_REPORT.md").write_text(build_structural_report(summary, metrics), encoding="utf-8")
    Path("reports/B6_3_TRACE_REPAIR_REPORT.md").write_text(build_trace_repair_report(summary, metrics), encoding="utf-8")


def base_context(episode: dict[str, Any], seed: int, policy_name: str) -> dict[str, Any]:
    return {"condition": episode["evaluator_ground_truth"]["condition"], "seed": seed, "policy_name": policy_name}


def empty_summary(condition: str, seed: int) -> dict[str, Any]:
    return {field: 0 for field in SUMMARY_FIELDS} | {
        "condition": condition,
        "seed": seed,
        "policy_name": "b63_policy",
        "ablation_name": "reference",
        "sample_count": 0,
        "no_sample_metric_count": 1,
        "invalid_metric_count": 1,
        "score_cap_reason": "no_samples",
    }


def build_metrics(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    b63_rows = [row for row in summary if row["policy_name"] == "b63_policy"]
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({row["condition"] for row in summary}),
        "b63_policy_mean_score": mean(b63_rows, "risk_constrained_score"),
        "mean_structural_necessity_score": mean([row for row in summary if row["policy_name"].startswith("b63_")], "structural_necessity_score"),
        "max_forbidden_reference_count": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "drop_by_ablation": drop_by_ablation(summary),
    }


def drop_by_ablation(summary: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        if row["ablation_name"] not in {"reference", "baseline", "reference_b62"}:
            grouped[row["ablation_name"]].append(row)
    return {name: mean(rows, "observed_performance_drop") for name, rows in sorted(grouped.items())}


def mechanism_for_policy(policy_name: str) -> str:
    if policy_name in {"b62_policy", "b63_policy"}:
        return "reference"
    if policy_name in BASELINE_NAMES:
        return "baseline"
    return POLICY_TO_ABLATION[policy_name]


def interpretation_for(policy_name: str, condition: str) -> str:
    ablation = POLICY_TO_ABLATION[policy_name]
    if ablation in EXPECTED_AFFECTED_SPLITS and condition in EXPECTED_AFFECTED_SPLITS[ablation]:
        return "expected_affected_split"
    if ablation == "baseline":
        return "baseline_comparison"
    return "control_or_reference"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_structural_report(summary: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    drops = metrics["drop_by_ablation"]
    lines = [
        "# B6.3 Structural Necessity Report",
        "",
        "## Purpose",
        "B6.3 tests whether B6.2 behavior causally depends on internal operational trace, history, feedback, delayed credit, and candidate-search structure.",
        "",
        "## Why B6.3",
        "B6.2 still had unresolved wrong_trace behavior, candidate-search-only hide_indirect_target, and unproven private trace necessity.",
        "",
        "## Ablation Table",
        "| ablation | mean observed drop | interpretation |",
        "| --- | ---: | --- |",
    ]
    for name, drop in drops.items():
        interpretation = "evidence_for_necessity" if drop >= 0.10 else "weak_or_no_necessity_evidence"
        lines.append(f"| {name} | {drop:.3f} | {interpretation} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "B6.3 can only support toy diagnostic structural necessity evidence if targeted ablations show specific performance collapses. It does not support real-world risk intelligence, safety certification, robotics ability, construction-site autonomy, or engineering deployment.",
            "",
        ]
    )
    return "\n".join(lines)


def build_trace_repair_report(summary: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    rows = [row for row in summary if row["policy_name"] == "b63_policy"]
    by_condition = {row["condition"]: row for row in rows}
    lines = [
        "# B6.3 Trace Repair Report",
        "",
        "## Trace Repair Result",
        "| condition | score | conflict detection | downgrade | repair attempt | inspect recovery | gain over state_only | gain over mask_only |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in ["wrong_trace", "wrong_trace_state_ambiguous", "hide_public_state_cue", "missing_trace", "ambiguous_trace", "low_confidence_trace"]:
        row = by_condition.get(condition)
        if not row:
            continue
        lines.append(
            f"| {condition} | {float(row['risk_constrained_score']):.3f} | {float(row['trace_conflict_detection_accuracy']):.3f} | "
            f"{float(row['trace_confidence_downgrade_rate']):.3f} | {float(row['trace_repair_attempt_rate']):.3f} | "
            f"{float(row['inspect_recovery_rate']):.3f} | {float(row['gain_over_state_only']):.3f} | {float(row['gain_over_mask_only']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "Trace repair is only structural evidence when it beats state_only/mask_only and degrades under trace/history/public-state ablations. If repair mainly follows public state cues, the claim remains narrow.",
            "",
        ]
    )
    return "\n".join(lines)
