from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, output: str) -> None:
    grouped = df.groupby("agent")[metric].mean().sort_values()
    ax = grouped.plot(kind="bar", figsize=(8, 4))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    plt.tight_layout()
    Path(output).parent.mkdir(exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/hardening_summary.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    plot_metric(df, "variable_delay_success", "figures/variable_delay_success.png")
    plot_metric(df, "false_delay_shortcut_rejection", "figures/false_shortcut_rejection.png")
    plot_metric(df, "multi_link_delay_edit_success", "figures/multi_link_edit_success.png")
    plot_metric(df, "hardening_gated_score", "figures/hardening_gated_score.png")


if __name__ == "__main__":
    main()
