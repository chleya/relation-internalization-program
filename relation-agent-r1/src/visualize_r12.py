from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_plot(rows: list[dict[str, str]], metric: str, output: str) -> None:
    agents = [row["agent"] for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(agents, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/r12_discovery_summary.csv")
    args = parser.parse_args()
    rows = read_rows(args.summary)
    bar_plot(rows, "unmarked_nuisance_rejection", "figures/r12_unmarked_nuisance_rejection.png")
    bar_plot(rows, "new_link_discovery", "figures/r12_new_link_discovery.png")
    bar_plot(rows, "adaptive_exploration", "figures/r12_adaptive_exploration.png")
    bar_plot(rows, "discovery_r12_gated_score", "figures/r12_discovery_score.png")


if __name__ == "__main__":
    main()
