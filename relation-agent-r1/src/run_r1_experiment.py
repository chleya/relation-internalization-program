from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .agents import RelationAgent, make_agent
from .metrics import evaluate_agent
from .report_r1 import build_claims, build_limitations, build_report, build_self_audit


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


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    relation_dump: dict[str, Any] = {}
    for agent_name in config["agents"]:
        for seed in config["seeds"]:
            agent = make_agent(agent_name, int(seed))
            metrics = evaluate_agent(agent, int(seed), config)
            records.append({"agent": agent_name, "seed": int(seed), **metrics})
            if isinstance(agent, RelationAgent):
                relation_dump[f"{agent_name}_seed{seed}"] = agent.describe_relations()

    summary = []
    for agent_name in config["agents"]:
        rows = [row for row in records if row["agent"] == agent_name]
        summary.append(
            {
                "agent": agent_name,
                "n": len(rows),
                "action_success": mean(row["action_success"] for row in rows),
                "ood_action_success": mean(row["ood_action_success"] for row in rows),
                "relation_recovery": mean(row["relation_recovery"] for row in rows),
                "counterfactual_accuracy": mean(row["counterfactual_accuracy"] for row in rows),
                "edit_success": mean(row["edit_success"] for row in rows),
                "active_exploration": mean(row["active_exploration"] for row in rows),
                "gated_r1_score": mean(row["gated_r1_score"] for row in rows),
            }
        )

    write_csv("results/r1_records.csv", records)
    write_csv("results/r1_summary.csv", summary)
    Path("results").mkdir(exist_ok=True)
    Path("results/r1_relations.json").write_text(json.dumps(relation_dump, indent=2), encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/R1_AGENT_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    Path("reports/R1_SELF_AUDIT.md").write_text(build_self_audit(), encoding="utf-8")
    Path("reports/R1_CLAIMS.md").write_text(build_claims(), encoding="utf-8")
    Path("reports/R1_LIMITATIONS.md").write_text(build_limitations(), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r1.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['agent']}: gated_r1_score={row['gated_r1_score']:.3f}")


if __name__ == "__main__":
    main()

