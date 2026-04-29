from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .partial_metrics import evaluate_r2_1_agent, r2_1_gated_score
from .report_r1 import markdown_table


METRICS = [
    "critical_missing_inspection_recall",
    "noncritical_missing_no_inspect_rate",
    "inspection_precision",
    "unnecessary_inspection_rate",
    "unsafe_automation_rate",
    "conflict_localization_accuracy",
    "cost_adjusted_success",
    "non_oracle_inspection_update_accuracy",
]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict[str, Any]], baseline: float) -> str:
    return "\n".join(
        [
            "# R2.1 Reviewer Hardening Report",
            "",
            "## 1. Motivation",
            "",
            "R2.1 tests whether inspection is relation-specific rather than a blanket response to any unknown field.",
            "",
            "## 2. Tests",
            "",
            "- Non-critical missingness should not force inspection.",
            "- Critical missingness should trigger inspection when the relation chain is unverifiable.",
            "- Mixed observability should allow action when an observed downstream path is sufficient.",
            "- Conflict audits must localize relation links.",
            "- Inspection has cost and a limited budget.",
            "- Inspection reveals one selected field, not the full true state.",
            "- The `missing_always_inspect` baseline is compared under cost-adjusted success and inspection precision.",
            "",
            "## 3. Results",
            "",
            markdown_table(rows),
            "",
            "## 4. Baseline Margin",
            "",
            f"`missing_always_inspect` cost_adjusted_success: {baseline:.3f}. A passing agent must exceed this by at least 0.100.",
            "",
            "## 5. Interpretation",
            "",
            "A passing R2.1 agent inspects because a specific physical relation link cannot be verified, not because any field is unknown. The hardening target is inspection precision under uncertainty, not maximal conservatism.",
            "",
            "## 6. Boundary",
            "",
            "R2.1 remains a toy diagnostic. It is not real slope monitoring, not calibrated inspection policy, not adversarial missingness robustness, and not deployment-ready engineering AI.",
            "",
        ]
    )


def build_self_audit(rows: list[dict[str, Any]]) -> str:
    target = [row for row in rows if row["agent"] == "relation_specific_uncertainty_agent"]
    score = target[0]["r2_1_gated_score"] if target else 0.0
    return "\n".join(
        [
            "# R2.1 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Separates relation-specific uncertainty from blanket inspection.",
            "- Adds non-critical missingness controls.",
            "- Adds inspection cost and a limited inspection budget.",
            "- Requires conflict localization for named relation links.",
            "- Makes inspection non-oracle by revealing only one selected field.",
            "",
            "## Remaining Weaknesses",
            "",
            "- This remains a toy diagnostic. It is not real slope monitoring, not calibrated inspection policy, not adversarial missingness robustness, and not deployment-ready engineering AI.",
            "- Missingness, conflicts, and inspection noise are synthetic.",
            "- The policy is hand-shaped for the diagnostic relation graph.",
            "- Inspection still uses a simple field-selection interface.",
            "",
            "## Current R2.1 Result",
            "",
            f"- relation_specific_uncertainty_agent r2_1_gated_score: {score:.3f}",
            "",
        ]
    )


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for agent_name in config["agents"]:
        for seed in config["seeds"]:
            records.append({"agent": agent_name, "seed": int(seed), **evaluate_r2_1_agent(agent_name, int(seed), config)})

    baseline_records = [row for row in records if row["agent"] == "missing_always_inspect"]
    baseline = mean(row["cost_adjusted_success"] for row in baseline_records)

    summary = []
    for agent_name in config["agents"]:
        rows = [row for row in records if row["agent"] == agent_name]
        metrics = {name: mean(row[name] for row in rows) for name in METRICS}
        metrics["r2_1_gated_score"] = r2_1_gated_score(metrics, config["gates"], baseline)
        summary.append({"agent": agent_name, "n": len(rows), **metrics})

    write_csv("results/r2_1_records.csv", records)
    write_csv("results/r2_1_summary.csv", summary)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/R2_1_HARDENING_REPORT.md").write_text(build_report(summary, baseline), encoding="utf-8")
    Path("reports/R2_1_SELF_AUDIT.md").write_text(build_self_audit(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r2_1_hardening.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['agent']}: r2_1_gated_score={row['r2_1_gated_score']:.3f}")


if __name__ == "__main__":
    main()
