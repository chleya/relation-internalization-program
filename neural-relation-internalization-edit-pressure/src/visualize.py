from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_plot(rows: list[dict[str, str]], metrics: list[str], output: str) -> None:
    models = [row["model"] for row in rows]
    width = 0.8 / len(metrics)
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, metric in enumerate(metrics):
        ax.bar([pos + i * width for pos in x], [float(row[metric]) for row in rows], width=width, label=metric)
    ax.set_xticks([pos + width * (len(metrics) - 1) / 2 for pos in x])
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(output).parent.mkdir(exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    args = parser.parse_args()
    rows = read_rows(args.summary)
    bar_plot(rows, ["gated_internalization_score"], "figures/gated_scores.png")
    bar_plot(rows, ["ood_accuracy", "shortcut_rejection_accuracy"], "figures/ood_shortcut.png")
    bar_plot(rows, ["reversal_adaptation_accuracy"], "figures/reversal_adaptation.png")
    bar_plot(rows, ["table_alignment", "table_edit_success", "edit_locality"], "figures/table_extraction.png")
    bar_plot(rows, ["relation_subspace_drop", "nuisance_subspace_drop"], "figures/subspace_drops.png")


if __name__ == "__main__":
    main()
