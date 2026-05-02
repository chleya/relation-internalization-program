from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .hard_remap_env import CONDITIONS, make_b641_episode
from .hard_remap_metrics import RECORD_FIELDS, SUMMARY_FIELDS, score_output, summarize_policy
from .hard_remap_policy import POLICY_NAMES, audit_policy_integrity, policy_for


def make_datasets(config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    section = config.get("b6_4_1", {})
    n = int(section.get("episodes_per_condition", 6))
    conditions = list(section.get("conditions", CONDITIONS))
    return {
        condition: [make_b641_episode(config, seed + cidx * 1000 + idx, condition) for idx in range(n)]
        for cidx, condition in enumerate(conditions)
    }


def run_b6_4_1_transfer_hardening(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    datasets = make_datasets(config, seed)
    records: list[dict[str, Any]] = []
    raw: dict[tuple[str, str], list[dict[str, Any]]] = {}
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
                out = policy(episode, config)
                scored = score_output(episode, out)
                row = {"condition": condition, "seed": seed, "policy_name": policy_name, **scored}
                rows.append(row)
                records.append(row)
            raw[(condition, policy_name)] = rows
    clean_score = mean(raw.get(("clean_reference", "b64_1_transfer_policy"), []), "hard_transfer_score")
    summary: list[dict[str, Any]] = []
    for condition in datasets:
        baselines = {policy_name: mean(raw.get((condition, policy_name), []), "hard_transfer_score") for policy_name in POLICY_NAMES}
        for policy_name in POLICY_NAMES:
            summary.append(
                summarize_policy(
                    raw.get((condition, policy_name), []),
                    {"condition": condition, "seed": seed, "policy_name": policy_name},
                    baselines,
                    audits[condition],
                    clean_score,
                )
            )
    metrics = build_metrics(summary, records)
    return summary, records, metrics


def build_metrics(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    b641 = [row for row in summary if row["policy_name"] == "b64_1_transfer_policy"]
    shortcut_equivalent = [
        row["condition"]
        for row in b641
        if row["condition"] != "clean_reference" and float(row["hard_baseline_transfer_gap"]) <= 0.05
    ]
    hard_gap = avg([float(row["hard_baseline_transfer_gap"]) for row in b641 if row["condition"] != "clean_reference"])
    shortcut_removed = avg([float(row["shortcut_removed_score"]) for row in b641 if row["condition"] != "clean_reference"])
    generalization = avg([float(row["hard_transfer_score"]) for row in b641 if row["condition"] != "clean_reference"])
    anti_overfit = avg([float(row["anti_overfit_score"]) for row in b641 if row["condition"] != "clean_reference"])
    integrity = 1.0 if max((int(row["forbidden_reference_count"]) for row in summary), default=0) == 0 and sum(int(row["invalid_metric_count"]) for row in summary) == 0 else 0.0
    evidence = 0.30 * max(0.0, hard_gap) + 0.25 * shortcut_removed + 0.20 * generalization + 0.15 * anti_overfit + 0.10 * integrity
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({row["condition"] for row in summary}),
        "hard_transfer_score": generalization,
        "hard_transfer_drop": avg([float(row["hard_transfer_drop"]) for row in b641 if row["condition"] != "clean_reference"]),
        "hard_baseline_transfer_gap": hard_gap,
        "hard_oracle_gap": avg([float(row["hard_oracle_gap"]) for row in b641 if row["condition"] != "clean_reference"]),
        "anti_overfit_score": anti_overfit,
        "shortcut_removed_score": shortcut_removed,
        "shortcut_equivalent_hard_remap_count": len(shortcut_equivalent),
        "shortcut_equivalent_hard_remaps": shortcut_equivalent,
        "transfer_evidence_strength": evidence,
        "forbidden_reference_count_max": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "no_sample_metric_count_total": sum(int(row["no_sample_metric_count"]) for row in summary),
        "remap_leakage_count_total": sum(int(row["remap_leakage_count"]) for row in summary),
        "submit_ready_as_diagnostic": True,
    }


def write_b6_4_1_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b6_4_1_transfer_hardening_summary.csv"), summary, SUMMARY_FIELDS)
    write_csv(Path("results/b6_4_1_transfer_hardening_records.csv"), records, RECORD_FIELDS)
    Path("results/b6_4_1_transfer_hardening_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_4_1_TRANSFER_HARDENING_REPORT.md").write_text(build_report(metrics), encoding="utf-8")


def build_report(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B6.4.1 Transfer Hardening",
            "",
            "## Purpose",
            "B6.4.1 hardens B6.4 first-pass remaps that were shortcut-explainable.",
            "",
            "## Results",
            f"- hard_transfer_score = {float(metrics['hard_transfer_score']):.3f}",
            f"- hard_baseline_transfer_gap = {float(metrics['hard_baseline_transfer_gap']):.3f}",
            f"- shortcut_equivalent_hard_remap_count = {int(metrics['shortcut_equivalent_hard_remap_count'])}",
            f"- transfer_evidence_strength = {float(metrics['transfer_evidence_strength']):.3f}",
            "",
            "## Claim Boundary",
            "B6.4.1 supports only toy-to-toy hard-remap diagnostic evidence. It does not support real-world generalization, robotics, construction-site autonomy, safety certification, or deployable control.",
            "",
            "combined_remap_hard remains limited when its oracle gap is non-trivial. A lower combined score should be treated as a hard-transfer limitation, not hidden by the aggregate score.",
            "",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
