from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_plot(rows: list[dict[str, str]], metrics: list[str], output: str, ylabel: str) -> None:
    agents = [row["agent"] for row in rows]
    x = range(len(agents))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, metric in enumerate(metrics):
        values = [float(row[metric]) for row in rows]
        positions = [pos + i * width for pos in x]
        ax.bar(positions, values, width=width, label=metric)
    ax.set_xticks([pos + width * (len(metrics) - 1) / 2 for pos in x])
    ax.set_xticklabels(agents, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/r2_1_summary.csv")
    args = parser.parse_args()
    rows = read_rows(args.summary)
    bar_plot(rows, ["r2_1_gated_score"], "figures/r2_1_gate_scores.png", "gated score")
    bar_plot(rows, ["inspection_precision", "critical_missing_inspection_recall"], "figures/inspection_precision_recall.png", "score")
    bar_plot(rows, ["unsafe_automation_rate", "unnecessary_inspection_rate"], "figures/unsafe_vs_unnecessary.png", "rate")
    bar_plot(rows, ["cost_adjusted_success"], "figures/cost_adjusted_success.png", "cost-adjusted success")


if __name__ == "__main__":
    main()
