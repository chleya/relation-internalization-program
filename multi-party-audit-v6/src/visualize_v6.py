from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_plot(rows: list[dict[str, str]], metric: str, output: str) -> None:
    resolvers = [row["resolver"] for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(resolvers, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/v6_audit_summary.csv")
    args = parser.parse_args()
    rows = read_rows(args.summary)
    bar_plot(rows, "gated_v6_score", "figures/v6_gated_score.png")
    bar_plot(rows, "disagreement_detection", "figures/v6_disagreement_detection.png")
    bar_plot(rows, "no_auto_resolution", "figures/v6_no_auto_resolution.png")
    bar_plot(rows, "minority_risk_preservation", "figures/v6_minority_risk_preservation.png")


if __name__ == "__main__":
    main()

