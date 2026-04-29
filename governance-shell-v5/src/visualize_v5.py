from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_plot(rows: list[dict[str, str]], metric: str, output: str) -> None:
    shells = [row["shell"] for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(shells, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/v5_governance_summary.csv")
    args = parser.parse_args()
    rows = read_rows(args.summary)
    bar_plot(rows, "gated_v5_score", "figures/v5_gated_score.png")
    bar_plot(rows, "approval_gate_enforcement", "figures/v5_gate_enforcement.png")
    bar_plot(rows, "replay_consistency", "figures/v5_replay_consistency.png")
    bar_plot(rows, "responsibility_traceability", "figures/v5_responsibility_traceability.png")


if __name__ == "__main__":
    main()

