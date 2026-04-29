from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def bar_with_points(df: pd.DataFrame, columns: list[str], output: Path) -> None:
    means = df[columns].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    means.plot(kind="bar", ax=ax, color="#4c78a8")
    for idx, column in enumerate(columns):
        ax.scatter([idx] * len(df), df[column], color="#f58518", s=28, zorder=3)
    ax.set_ylim(0, max(1.0, float(df[columns].max().max()) + 0.05))
    ax.set_ylabel("score")
    ax.set_title("Neural relation probe metrics")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def grouped_bars(df: pd.DataFrame, columns: list[str], output: Path) -> None:
    grouped = df.groupby("train_mode")[columns].mean()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylim(0, max(1.0, float(grouped.max().max()) + 0.05))
    ax.set_ylabel("mean score")
    ax.set_title("Neural relation probe by training mode")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    if "gated_extraction_score" in df.columns:
        columns = [
            "table_ood_accuracy",
            "table_spurious_attack_accuracy",
            "table_relation_alignment",
            "gated_extraction_score",
        ]
        output = Path("figures/extraction_modes.png") if "train_mode" in df.columns else Path("figures/extraction_summary.png")
        if "train_mode" in df.columns and df["train_mode"].nunique() > 1:
            grouped_bars(df, columns, output)
        else:
            bar_with_points(df, columns, output)
        return

    columns = [
        "ood_accuracy",
        "spurious_attack_accuracy",
        "probe_selectivity",
        "relation_subspace_drop",
        "nuisance_subspace_drop",
        "gated_neural_relation_score",
    ]
    if "train_mode" in df.columns and df["train_mode"].nunique() > 1:
        grouped_bars(df, columns, Path("figures/neural_probe_modes.png"))
    else:
        bar_with_points(df, columns, Path("figures/neural_probe_summary.png"))


if __name__ == "__main__":
    main()
