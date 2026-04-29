from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .metrics_r12 import evaluate_discovery_agent
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
            "# R1.2 Discovery Report",
            "",
            "## 1. Motivation",
            "",
            "R1.2 reduces R1.1 scaffolding by replacing hand-written TRUE_LINK candidates with relation candidates enumerated from observed transitions.",
            "",
            "## 2. Tests",
            "",
            "- Unmarked nuisance rejection.",
            "- New process link discovery outside the original TRUE_LINKS set.",
            "- Adaptive exploration instead of a fixed exploration schedule.",
            "- Discovered relation precision under nuisance pressure.",
            "- Relation-guided action success.",
            "",
            "## 3. Results",
            "",
            markdown_table(rows),
            "",
            "## 4. Interpretation",
            "",
            "The discovery agent must learn usable relations from transition evidence rather than receiving the original true-link table. Negative controls should still fail the gated score.",
            "",
            "## 5. Boundary",
            "",
            "R1.2 is not unrestricted causal discovery. It still uses a small process-variable schema, deterministic dynamics, and a toy action space.",
            "",
        ]
    )


def build_self_audit(rows: list[dict[str, Any]]) -> str:
    discovery_rows = [row for row in rows if row["agent"] == "discovery_relation_agent"]
    score = discovery_rows[0]["discovery_r12_gated_score"] if discovery_rows else 0.0
    return "\n".join(
        [
            "# R1.2 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Removes direct dependence on the hand-written TRUE_LINK candidate table for the positive agent.",
            "- Tests an unmarked nuisance setting rather than passing explicit nuisance labels into the agent.",
            "- Tests discovery of a synthetic new process link.",
            "- Uses adaptive low-coverage exploration instead of a fixed scripted schedule.",
            "",
            "## Remaining Weaknesses",
            "",
            "- The discovery agent still uses a process-variable schema.",
            "- The environment is still small and deterministic.",
            "- New-link discovery is synthetic and simple.",
            "- The agent does not perform open-ended causal discovery.",
            "- This is not an LLM replacement or real engineering intelligence.",
            "",
            "## Current R1.2 Result",
            "",
            f"- discovery_relation_agent discovery_r12_gated_score: {score:.3f}",
            "",
        ]
    )


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for agent_name in config["agents"]:
        for seed in config["seeds"]:
            records.append({"agent": agent_name, "seed": int(seed), **evaluate_discovery_agent(agent_name, int(seed), config)})

    metric_names = [
        "unmarked_nuisance_rejection",
        "new_link_discovery",
        "adaptive_exploration",
        "discovered_relation_precision",
        "discovery_action_success",
        "discovery_r12_gated_score",
    ]
    summary = []
    for agent_name in config["agents"]:
        rows = [row for row in records if row["agent"] == agent_name]
        summary.append({"agent": agent_name, "n": len(rows), **{name: mean(row[name] for row in rows) for name in metric_names}})

    write_csv("results/r12_discovery_records.csv", records)
    write_csv("results/r12_discovery_summary.csv", summary)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/R1_2_DISCOVERY_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    Path("reports/R1_2_SELF_AUDIT.md").write_text(build_self_audit(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r12_discovery.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['agent']}: discovery_r12_gated_score={row['discovery_r12_gated_score']:.3f}")


if __name__ == "__main__":
    main()
