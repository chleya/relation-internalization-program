from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .metrics_r11 import evaluate_hardening_agent
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


def build_hardening_report(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# R1.1 Hardening Report",
            "",
            "## 1. Motivation",
            "",
            "R1.1 tests whether the R1 relation agent is more than a hand-fitted process learner.",
            "",
            "## 2. Attacks",
            "",
            "- Hidden nuisance/confounder candidates.",
            "- Process-rule reversal after base training.",
            "- Intervention cost tradeoff.",
            "- Expanded candidate relation set with irrelevant links.",
            "- Active-exploration ablation.",
            "",
            "## 3. Results",
            "",
            markdown_table(rows),
            "",
            "## 4. Interpretation",
            "",
            "A passing agent must keep relation-guided action under nuisance shifts, adapt to a changed process rule, avoid unnecessary costly interventions, reject irrelevant candidate links, and actually perform active exploration.",
            "",
            "## 5. Boundary",
            "",
            "R1.1 remains a toy diagnostic. It is not unrestricted causal discovery, not an LLM replacement, and not real engineering competence.",
            "",
        ]
    )


def build_self_audit(rows: list[dict[str, Any]]) -> str:
    relation_rows = [row for row in rows if row["agent"] == "relation_agent"]
    relation_score = relation_rows[0]["hardening_r11_gated_score"] if relation_rows else 0.0
    return "\n".join(
        [
            "# R1.1 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Adds nuisance/confounder pressure.",
            "- Tests adaptation after a process-rule reversal.",
            "- Tests whether costly interventions are avoided when risk is already low.",
            "- Expands the candidate relation set with irrelevant links.",
            "- Adds a no-exploration negative control.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Candidate links are still finite and mostly predefined.",
            "- Nuisance features are explicitly marked for the hardened relation agent.",
            "- The environment is still small and deterministic.",
            "- Process reversal is simple and synthetic.",
            "- This is not a replacement for LLM-scale intelligence.",
            "",
            "## Current R1.1 Result",
            "",
            f"- relation_agent hardening_r11_gated_score: {relation_score:.3f}",
            "",
            "## Boundary",
            "",
            "This remains a toy active relation-learning diagnostic, not real engineering AI and not unrestricted causal discovery.",
            "",
        ]
    )


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for agent_name in config["agents"]:
        for seed in config["seeds"]:
            metrics = evaluate_hardening_agent(agent_name, int(seed), config)
            records.append({"agent": agent_name, "seed": int(seed), **metrics})

    metric_names = [
        "hidden_confounder_rejection",
        "reversal_adaptation",
        "cost_tradeoff_success",
        "candidate_expansion_precision",
        "active_discovery_score",
        "hardening_r11_gated_score",
    ]
    summary = []
    for agent_name in config["agents"]:
        rows = [row for row in records if row["agent"] == agent_name]
        summary.append({"agent": agent_name, "n": len(rows), **{name: mean(row[name] for row in rows) for name in metric_names}})

    write_csv("results/r11_hardening_records.csv", records)
    write_csv("results/r11_hardening_summary.csv", summary)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/R1_1_HARDENING_REPORT.md").write_text(build_hardening_report(summary), encoding="utf-8")
    Path("reports/R1_1_SELF_AUDIT.md").write_text(build_self_audit(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r11_hardening.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['agent']}: hardening_r11_gated_score={row['hardening_r11_gated_score']:.3f}")


if __name__ == "__main__":
    main()
