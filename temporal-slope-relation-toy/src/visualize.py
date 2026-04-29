from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/temporal_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    cols = [
        "temporal_ood_success",
        "delayed_counterfactual_accuracy",
        "delay_edit_success",
        "surface_shortcut_rejection",
        "temporal_audit_score",
        "gated_temporal_score",
    ]
    grouped = df.groupby("agent")[cols].mean()
    Path("figures").mkdir(exist_ok=True)
    ax = grouped.plot(kind="bar", figsize=(11, 4.8))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean score")
    ax.set_title("V2 temporal relation diagnostics")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig("figures/temporal_scores.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
