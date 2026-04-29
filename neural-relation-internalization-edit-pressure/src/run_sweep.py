from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .evaluate import evaluate_all
from .train import train_model


METRICS = [
    "train_accuracy",
    "ood_accuracy",
    "spurious_attack_accuracy",
    "shortcut_rejection_accuracy",
    "reversal_adaptation_accuracy",
    "counterfactual_consistency",
    "probe_relation_accuracy",
    "probe_nuisance_accuracy",
    "probe_selectivity",
    "relation_subspace_drop",
    "nuisance_subspace_drop",
    "table_alignment",
    "table_ood_accuracy",
    "table_spurious_attack_accuracy",
    "table_edit_success",
    "edit_locality",
    "gated_internalization_score",
]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Auto Report",
        "",
        "## Results",
        "",
        "| model | gated | ood | shortcut | reversal | table | edit | locality | rel_drop | nuis_drop |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['gated_internalization_score']:.3f} | {row['ood_accuracy']:.3f} | "
            f"{row['shortcut_rejection_accuracy']:.3f} | {row['reversal_adaptation_accuracy']:.3f} | "
            f"{row['table_alignment']:.3f} | {row['table_edit_success']:.3f} | {row['edit_locality']:.3f} | "
            f"{row['relation_subspace_drop']:.3f} | {row['nuisance_subspace_drop']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This is a toy diagnostic for neural relation internalization under edit pressure. A non-zero gated score means all configured gates passed; a zero score means at least one structural gate failed.",
    ]
    return "\n".join(lines)


def build_self_audit() -> str:
    return "\n".join(
        [
            "# Self-Audit",
            "",
            "## What this experiment improves",
            "- Moves from hand-written relation agents to neural relation emergence pressure.",
            "- Tests whether relation structure can be extracted from learned hidden states.",
            "- Tests editability and locality of extracted relations.",
            "- Tests whether relation subspace is behaviorally causal.",
            "",
            "## Remaining weaknesses",
            "- Toy world remains simple.",
            "- Edit pressure is still a designed training signal.",
            "- Extracted table is based on canonical probing.",
            "- Relation variables are known in the data generator.",
            "- This is not unrestricted causal discovery.",
            "- This is not proof of relation understanding in large neural models.",
            "",
            "## False positive risks",
            "- Model may learn canonical query patterns.",
            "- Table extraction may overfit nuisance averaging.",
            "- Edit-pressure architecture may impose too much relation-like structure.",
            "- Probe intervention may remove correlated features, not pure relation factors.",
            "",
            "## Required failure checks",
            "1. Pure prediction passes all gates.",
            "2. Edit-pressure model fails table edit.",
            "3. Extracted table is correct only on canonical queries.",
            "4. Relation subspace drop is small.",
            "5. Nuisance subspace drop is large.",
            "6. Edit locality is poor.",
            "",
        ]
    )


def run(config_path: str) -> list[dict[str, Any]]:
    sweep = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base_config = yaml.safe_load(Path(sweep["base_config"]).read_text(encoding="utf-8"))
    records = []
    for model_name in sweep["models"]:
        for seed in sweep["seeds"]:
            model = train_model(model_name, int(seed), base_config)
            records.append({"model": model_name, "seed": int(seed), **evaluate_all(model, base_config, int(seed), model_name)})
            print(f"{model_name} seed={seed} gated={records[-1]['gated_internalization_score']:.3f}")
    summary = []
    for model_name in sweep["models"]:
        rows = [row for row in records if row["model"] == model_name]
        summary.append({"model": model_name, "n": len(rows), **{metric: mean(row[metric] for row in rows) for metric in METRICS}})
    write_csv("results/records.csv", records)
    write_csv("results/summary.csv", summary)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/AUTO_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    Path("reports/SELF_AUDIT.md").write_text(build_self_audit(), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
