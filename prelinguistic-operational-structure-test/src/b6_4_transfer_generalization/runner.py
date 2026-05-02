from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .remap_configs import CONDITIONS, POLICY_NAMES
from .remap_env import make_b64_episode
from .transfer_metrics import RECORD_FIELDS, SUMMARY_FIELDS, score_output, summarize_policy
from .transfer_policy import audit_policy_integrity, policy_for


def make_datasets(config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    section = config.get("b6_4", {})
    n = int(section.get("episodes_per_condition", 6))
    conditions = list(section.get("conditions", CONDITIONS))
    return {
        condition: [make_b64_episode(config, seed + cidx * 1000 + idx, condition) for idx in range(n)]
        for cidx, condition in enumerate(conditions)
    }


def run_b6_4_transfer(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    datasets = make_datasets(config, seed)
    raw_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    audits: dict[str, dict[str, Any]] = {}

    for condition, episodes in datasets.items():
        audits[condition] = audit_policy_integrity(episodes[0], config) if episodes else {
            "forbidden_reference_count": 0,
            "poisoned_evaluator_invariance_pass": False,
            "policy_uses_model_input_only": False,
        }
        for policy_name in POLICY_NAMES:
            policy = policy_for(policy_name)
            rows = []
            for episode in episodes:
                output = policy(episode, config)
                scored = score_output(episode, output)
                row = {"condition": condition, "seed": seed, "policy_name": policy_name, **scored}
                rows.append(row)
                records.append(row)
            raw_rows[(condition, policy_name)] = rows

    clean_reference_score = mean_score(raw_rows.get(("clean_reference", "b64_transfer_policy"), []))
    summary: list[dict[str, Any]] = []
    for condition in datasets:
        baseline_scores = {policy: mean_score(raw_rows.get((condition, policy), [])) for policy in POLICY_NAMES}
        for policy_name in POLICY_NAMES:
            rows = raw_rows.get((condition, policy_name), [])
            summary.append(
                summarize_policy(
                    rows,
                    {"condition": condition, "seed": seed, "policy_name": policy_name},
                    baseline_scores,
                    audits[condition],
                    clean_reference_score,
                )
            )
    metrics = build_metrics(summary, records)
    return summary, records, metrics


def build_metrics(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    b64 = [row for row in summary if row["policy_name"] == "b64_transfer_policy"]
    by_condition = {row["condition"]: row for row in b64}
    transfer_scores = [float(row["transfer_score"]) for row in b64 if row["condition"] != "clean_reference"]
    baseline_gaps = [float(row["baseline_transfer_gap"]) for row in b64 if row["condition"] != "clean_reference"]
    mechanism_scores = [
        max(
            float(row["trace_repair_transfer_score"]),
            float(row["fallback_risk_transfer_score"]),
            float(row["delayed_credit_transfer_score"]),
            float(row["hidden_indirect_transfer_score"]),
            float(row["feedback_history_transfer_score"]),
        )
        for row in b64
        if row["condition"] != "clean_reference"
    ]
    transfer_score = avg(transfer_scores)
    baseline_gap = avg(baseline_gaps)
    mechanism_transfer = avg(mechanism_scores)
    anti_overfit = max(0.0, min(1.0, 0.5 + baseline_gap))
    integrity = 1.0 if max((int(row["forbidden_reference_count"]) for row in summary), default=0) == 0 and sum(int(row["invalid_metric_count"]) for row in summary) == 0 else 0.0
    b64_score = 0.30 * transfer_score + 0.25 * max(0.0, baseline_gap) + 0.20 * mechanism_transfer + 0.15 * anti_overfit + 0.10 * integrity
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({row["condition"] for row in summary}),
        "b64_policy_mean_transfer_score": avg([float(row["transfer_score"]) for row in b64]),
        "b6_4_transfer_score": b64_score,
        "transfer_score": transfer_score,
        "transfer_drop": avg([float(row["transfer_drop"]) for row in b64 if row["condition"] != "clean_reference"]),
        "anti_overfit_score": anti_overfit,
        "remap_generalization_score": transfer_score,
        "baseline_transfer_gap": baseline_gap,
        "oracle_gap": avg([float(row["oracle_gap"]) for row in b64 if row["condition"] != "clean_reference"]),
        "visual_remap_score": by_condition.get("visual_remap", {}).get("transfer_score", 0.0),
        "risk_cue_remap_score": by_condition.get("risk_cue_remap", {}).get("transfer_score", 0.0),
        "dynamics_remap_score": by_condition.get("dynamics_remap", {}).get("transfer_score", 0.0),
        "delay_profile_remap_score": by_condition.get("delay_profile_remap", {}).get("transfer_score", 0.0),
        "indirect_path_remap_score": by_condition.get("indirect_path_remap", {}).get("transfer_score", 0.0),
        "mask_visibility_remap_score": by_condition.get("mask_visibility_remap", {}).get("transfer_score", 0.0),
        "combined_remap_score": by_condition.get("combined_remap", {}).get("transfer_score", 0.0),
        "forbidden_reference_count_max": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "no_sample_metric_count_total": sum(int(row["no_sample_metric_count"]) for row in summary),
        "poisoned_evaluator_invariance_all_pass": all(bool(row["poisoned_evaluator_invariance_pass"]) for row in summary),
        "submit_ready_as_diagnostic": True,
    }


def write_b6_4_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b6_4_transfer_summary.csv"), summary, SUMMARY_FIELDS)
    write_csv(Path("results/b6_4_transfer_records.csv"), records, RECORD_FIELDS)
    Path("results/b6_4_transfer_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_4_TRANSFER_GENERALIZATION_REPORT.md").write_text(build_report(metrics), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B6.4 Transfer and Anti-Overfit Generalization",
            "",
            "## Purpose",
            "B6.4 tests toy-to-toy transfer and anti-overfit generalization. It does not enter B7 or make real-world control claims.",
            "",
            "## Remap Conditions",
            "- visual_remap",
            "- risk_cue_remap",
            "- dynamics_remap",
            "- delay_profile_remap",
            "- indirect_path_remap",
            "- mask_visibility_remap",
            "- combined_remap",
            "",
            "## Results",
            f"- b6_4_transfer_score = {float(metrics['b6_4_transfer_score']):.3f}",
            f"- transfer_score = {float(metrics['transfer_score']):.3f}",
            f"- transfer_drop = {float(metrics['transfer_drop']):.3f}",
            f"- baseline_transfer_gap = {float(metrics['baseline_transfer_gap']):.3f}",
            f"- oracle_gap = {float(metrics['oracle_gap']):.3f}",
            f"- combined_remap_score = {float(metrics['combined_remap_score']):.3f}",
            "",
            "## Interpretation",
            "B6.4 is a transfer diagnostic. Remap success supports only toy-to-toy operational structure transfer evidence. Remap failures indicate split-specific or shortcut-explainable behavior.",
            "",
            "High transfer scores must be interpreted against baseline separation. If state_only, mask_only, or trace_only match the transfer policy on a remap, that split remains shortcut-explainable and should not be treated as strong operational transfer evidence.",
            "",
            "## Claim Boundary",
            "B6.4 supports only toy-to-toy transfer evidence. It does not support real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.",
            "",
        ]
    )


def mean_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get("transfer_score", 0.0)) for row in rows) / len(rows)


def avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
