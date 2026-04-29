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
    parser.add_argument("--summary", default="results/v61_hardening_summary.csv")
    args = parser.parse_args()
    rows = read_rows(args.summary)
    bar_plot(rows, "fake_evidence_rejection", "figures/v61_fake_evidence_rejection.png")
    bar_plot(rows, "hidden_auto_resolution_rejection", "figures/v61_hidden_auto_resolution_rejection.png")
    bar_plot(rows, "minority_risk_log_integrity", "figures/v61_minority_risk_log_integrity.png")
    bar_plot(rows, "hardening_v61_gated_score", "figures/v61_hardening_score.png")


if __name__ == "__main__":
    main()

