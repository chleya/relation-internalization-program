import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .run_experiment import run
from .utils import load_yaml


SUMMARY_COLUMNS = [
    "agent",
    "seed",
    "base_reward",
    "base_success",
    "reversal_adaptation_steps",
    "reversal_post_reward",
    "frozen_reversal_success",
    "ood_success",
    "spurious_robustness",
    "spurious_resource_accuracy",
    "counterfactual_accuracy",
    "edit_success",
    "edit_resource_success",
    "edit_resource_locality",
    "edit_locality",
    "edit_reversal_success",
    "relation_resource_accuracy",
    "relation_shuffled_resource_accuracy",
    "relation_shuffle_drop",
    "relation_table_coverage",
    "relation_table_accuracy",
    "relation_table_alignment",
    "internalization_score",
    "gated_internalization_score",
]


def write_report(summary: pd.DataFrame) -> None:
    Path("reports").mkdir(exist_ok=True)
    means = summary.groupby("agent")[SUMMARY_COLUMNS[2:]].mean(numeric_only=True).sort_values(
        "gated_internalization_score", ascending=False
    )
    lines = [
        "# Auto Report: Relation Internalization Test",
        "",
        "## Summary",
        "",
        "```text",
        means.round(3).to_string(),
        "```",
        "",
        "## Failure Checks",
        "",
    ]
    relation = means.loc["relation"] if "relation" in means.index else None
    if relation is not None:
        for col in [
            "ood_success",
            "spurious_robustness",
            "spurious_resource_accuracy",
            "counterfactual_accuracy",
            "edit_success",
            "edit_resource_success",
            "edit_locality",
            "edit_resource_locality",
            "edit_reversal_success",
            "relation_shuffle_drop",
            "relation_table_alignment",
            "gated_internalization_score",
        ]:
            best_baseline = means.drop(index="relation", errors="ignore")[col].max()
            status = "PASS" if relation[col] > best_baseline else "FAIL"
            lines.append(f"- {status}: relation {col}={relation[col]:.3f}, best baseline={best_baseline:.3f}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report is generated from the minimal experiment. Treat positive results as evidence for this testbed only, not as a broad theory claim.",
        ]
    )
    Path("reports/auto_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep.yaml")
    args = parser.parse_args()
    sweep = load_yaml(args.config)
    rows = []
    for agent in tqdm(sweep["agents"], desc="agents"):
        for seed in sweep["seeds"]:
            rows.append(run(agent, sweep["base_config"], int(seed)))
    summary = pd.DataFrame(rows)
    summary = summary[[c for c in SUMMARY_COLUMNS if c in summary.columns] + [c for c in summary.columns if c not in SUMMARY_COLUMNS]]
    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/summary.csv", index=False)
    write_report(summary)
    print(summary[SUMMARY_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
