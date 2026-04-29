from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .partial_metrics import evaluate_r3_agent, r3_gated_score
from .report_r1 import markdown_table


METRICS = [
    "inspection_target_accuracy",
    "information_gain_efficiency",
    "budgeted_safe_action_rate",
    "unsafe_automation_rate",
    "overinspection_rate",
    "sequential_update_accuracy",
    "cost_adjusted_success",
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


def build_report(rows: list[dict[str, Any]], first_missing: float, random_inspect: float) -> str:
    return "\n".join(
        [
            "# R3 Active Inspection Selection Report",
            "",
            "## 1. Motivation",
            "",
            "R3 tests whether an agent can choose the most informative field to inspect under multi-field missingness, noise, cost, and limited budget.",
            "",
            "## 2. Tests",
            "",
            "- Multiple critical and noncritical fields may be unknown at the same time.",
            "- The agent must select an inspection target.",
            "- The agent may inspect one field, update the observation, then decide again.",
            "- Inspection value is rule-based: chain position plus action relevance plus conflict resolution minus noncritical penalty.",
            "- Overinspection and unsafe automation are penalized under budget.",
            "",
            "## 3. Results",
            "",
            markdown_table(rows),
            "",
            "## 4. Baseline Margins",
            "",
            f"`first_missing_inspect` cost_adjusted_success: {first_missing:.3f}.",
            f"`random_inspect_field` cost_adjusted_success: {random_inspect:.3f}.",
            "",
            "## 5. Interpretation",
            "",
            "A passing R3 agent should inspect the field with highest expected decision value, not merely inspect the first or any missing field.",
            "",
        ]
    )


def build_self_audit(rows: list[dict[str, Any]]) -> str:
    target = [row for row in rows if row["agent"] == "active_inspection_agent"]
    score = target[0]["r3_gated_score"] if target else 0.0
    return "\n".join(
        [
            "# R3 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Adds multi-field missingness.",
            "- Requires explicit inspection target selection.",
            "- Allows sequential inspection with one-field updates.",
            "- Adds budgeted cost and overinspection penalties.",
            "- Compares against first-missing, random-field, and blanket-inspection baselines.",
            "",
            "## Remaining Weaknesses",
            "",
            "- This remains a toy diagnostic. It is not real field monitoring, not learned sensor reliability calibration, not adversarial inspection policy, and not deployment-ready engineering AI.",
            "- The inspection value function is rule-based.",
            "- Missingness and noise are synthetic.",
            "- The field graph is small and hand-specified.",
            "",
            "## Current R3 Result",
            "",
            f"- active_inspection_agent r3_gated_score: {score:.3f}",
            "",
        ]
    )


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for agent_name in config["agents"]:
        for seed in config["seeds"]:
            records.append({"agent": agent_name, "seed": int(seed), **evaluate_r3_agent(agent_name, int(seed), config)})

    first_missing = mean(row["cost_adjusted_success"] for row in records if row["agent"] == "first_missing_inspect")
    random_inspect = mean(row["cost_adjusted_success"] for row in records if row["agent"] == "random_inspect_field")

    summary = []
    for agent_name in config["agents"]:
        rows = [row for row in records if row["agent"] == agent_name]
        metrics = {name: mean(row[name] for row in rows) for name in METRICS}
        metrics["r3_gated_score"] = r3_gated_score(metrics, config["gates"], first_missing, random_inspect)
        summary.append({"agent": agent_name, "n": len(rows), **metrics})

    write_csv("results/r3_records.csv", records)
    write_csv("results/r3_summary.csv", summary)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/R3_ACTIVE_INSPECTION_REPORT.md").write_text(build_report(summary, first_missing, random_inspect), encoding="utf-8")
    Path("reports/R3_SELF_AUDIT.md").write_text(build_self_audit(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r3_active_inspection.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['agent']}: r3_gated_score={row['r3_gated_score']:.3f}")


if __name__ == "__main__":
    main()
