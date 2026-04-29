from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, output: str) -> None:
    grouped = df.groupby("agent")[metric].mean().sort_values()
    ax = grouped.plot(kind="bar", figsize=(9, 4))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    plt.tight_layout()
    Path(output).parent.mkdir(exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/v32_stress_summary.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    plot_metric(df, "long_missing_takeover_recall", "figures/v32_long_missing_takeover_recall.png")
    plot_metric(df, "correlated_failure_recall", "figures/v32_correlated_failure_recall.png")
    plot_metric(df, "multi_conflict_audit_score", "figures/v32_multi_conflict_audit_score.png")
    plot_metric(df, "stress_v32_gated_score", "figures/v32_stress_gated_score.png")


if __name__ == "__main__":
    main()

