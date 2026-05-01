from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b5_closed_loop_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "b5_closed_loop_score", "B5 closed-loop score", "figures/b5_closed_loop_scores.png")
    plot_pair(rows, "epistemic_value_alignment", "pragmatic_value_alignment", "Epistemic vs pragmatic", "figures/b5_epistemic_vs_pragmatic.png")
    plot_metric(rows, "trace_update_accuracy", "Trace update", "figures/b5_trace_update.png")
    plot_metric(rows, "post_inspection_intervention_accuracy", "Intervention after update", "figures/b5_intervention_after_update.png")
    plot_metric(rows, "feedback_revision_accuracy", "Feedback revision", "figures/b5_feedback_revision.png")
    plot_pair(rows, "closed_loop_gain_over_inspect_always", "closed_loop_gain_over_intervene_immediately", "Baseline comparison", "figures/b5_baseline_comparison.png")
    plot_metric(rows, "planning_budget_compliance", "Planning budget", "figures/b5_planning_budget.png")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_metric(rows: list[dict[str, str]], key: str, title: str, output: str) -> None:
    labels = [row["model"].replace("_model", "") for row in rows]
    values = [float(row.get(key, 0.0)) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#2f6f73")
    plt.ylim(0, 1.05)
    plt.ylabel(key)
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_pair(rows: list[dict[str, str]], left: str, right: str, title: str, output: str) -> None:
    labels = [row["model"].replace("_model", "") for row in rows]
    xs = range(len(rows))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar([x - width / 2 for x in xs], [float(row.get(left, 0.0)) for row in rows], width=width, label=left, color="#345995")
    plt.bar([x + width / 2 for x in xs], [float(row.get(right, 0.0)) for row in rows], width=width, label=right, color="#d65f5f")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.xticks(list(xs), labels, rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


if __name__ == "__main__":
    main()
