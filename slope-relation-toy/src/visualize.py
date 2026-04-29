from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    cols = [
        "ood_success",
        "spurious_attack_success",
        "counterfactual_accuracy",
        "edit_success",
        "relation_audit",
        "irrelevant_link_rejection",
        "noisy_observation_success",
        "review_score",
        "review_consistency",
        "gated_slope_score",
    ]
    grouped = df.groupby("agent")[cols].mean()
    Path("figures").mkdir(exist_ok=True)
    ax = grouped.plot(kind="bar", figsize=(10, 4.8))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean score")
    ax.set_title("Slope relation toy diagnostics")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig("figures/slope_scores.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
