from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .metrics_r2 import evaluate_partial_observability_agent
from .report_r1 import markdown_table


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


def build_report(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# R2 Partial Observability Report",
            "",
            "## 1. Motivation",
            "",
            "R2 tests whether an active relation agent can use learned relations when observations are missing, noisy, or conflicting.",
            "",
            "## 2. Tests",
            "",
            "- Partial observation success.",
            "- Inspection recall when critical relation nodes are unknown.",
            "- Unsafe automation rate.",
            "- Relation-specific uncertainty audit.",
            "- Noisy observation robustness.",
            "",
            "## 3. Results",
            "",
            markdown_table(rows),
            "",
            "## 4. Interpretation",
            "",
            "A passing agent must inspect before acting when key relation links cannot be verified, then act from the revealed state. Prediction or memory alone is not enough.",
            "",
            "## 5. Boundary",
            "",
            "R2 remains a toy diagnostic. It is not a real monitoring system, not engineering safety automation, and not unrestricted causal discovery.",
            "",
        ]
    )


def build_self_audit(rows: list[dict[str, Any]]) -> str:
    target = [row for row in rows if row["agent"] == "uncertainty_discovery_agent"]
    score = target[0]["partial_r2_gated_score"] if target else 0.0
    return "\n".join(
        [
            "# R2 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Adds missing observations.",
            "- Adds noisy/conflicting observations.",
            "- Requires inspect before unsafe automation under uncertainty.",
            "- Requires an audit naming the uncertain relation link.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Missingness and noise are synthetic.",
            "- Inspect reveals the true state immediately.",
            "- The process world remains small and deterministic.",
            "- The uncertainty policy is rule-based, not learned from deployment feedback.",
            "- This is not real engineering AI.",
            "",
            "## Current R2 Result",
            "",
            f"- uncertainty_discovery_agent partial_r2_gated_score: {score:.3f}",
            "",
        ]
    )


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for agent_name in config["agents"]:
        for seed in config["seeds"]:
            records.append({"agent": agent_name, "seed": int(seed), **evaluate_partial_observability_agent(agent_name, int(seed), config)})

    metric_names = [
        "partial_observation_success",
        "inspection_recall",
        "unsafe_action_rate",
        "uncertainty_audit_score",
        "noisy_observation_robustness",
        "partial_r2_gated_score",
    ]
    summary = []
    for agent_name in config["agents"]:
        rows = [row for row in records if row["agent"] == agent_name]
        summary.append({"agent": agent_name, "n": len(rows), **{name: mean(row[name] for row in rows) for name in metric_names}})

    write_csv("results/r2_partial_observability_records.csv", records)
    write_csv("results/r2_partial_observability_summary.csv", summary)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/R2_PARTIAL_OBSERVABILITY_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    Path("reports/R2_SELF_AUDIT.md").write_text(build_self_audit(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r2_partial_observability.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['agent']}: partial_r2_gated_score={row['partial_r2_gated_score']:.3f}")


if __name__ == "__main__":
    main()
