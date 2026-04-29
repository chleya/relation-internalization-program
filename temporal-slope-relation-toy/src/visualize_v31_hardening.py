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
    parser.add_argument("--summary", default="results/v31_hardening_summary.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    plot_metric(df, "takeover_overuse_control", "figures/v31_takeover_overuse_control.png")
    plot_metric(df, "conflicting_evidence_takeover", "figures/v31_conflicting_evidence_takeover.png")
    plot_metric(df, "audit_specificity", "figures/v31_audit_specificity.png")
    plot_metric(df, "hardening_v31_gated_score", "figures/v31_hardening_gated_score.png")


if __name__ == "__main__":
    main()

