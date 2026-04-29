import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def barplot(summary: pd.DataFrame, column: str, output: str, ylabel: str) -> None:
    means = summary.groupby("agent")[column].mean(numeric_only=True).sort_values()
    if column != "reversal_adaptation_steps":
        means = means.sort_values(ascending=False)
    errors = summary.groupby("agent")[column].std(numeric_only=True).reindex(means.index).fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    means.plot(kind="bar", yerr=errors, ax=ax, color="#4c78a8", capsize=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("agent")
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    Path("figures").mkdir(exist_ok=True)
    barplot(summary, "reversal_adaptation_steps", "figures/reversal_adaptation.png", "Reversal adaptation steps")
    barplot(summary, "ood_success", "figures/ood_success.png", "OOD success")
    if "spurious_robustness" in summary.columns:
        barplot(summary, "spurious_robustness", "figures/spurious_robustness.png", "Spurious attack success")
    if "relation_shuffle_drop" in summary.columns:
        barplot(summary, "relation_shuffle_drop", "figures/relation_shuffle_drop.png", "Relation shuffle drop")
    if "relation_table_alignment" in summary.columns:
        barplot(summary, "relation_table_alignment", "figures/relation_table_alignment.png", "Relation table alignment")
    summary["edit_score"] = (summary["edit_success"] + summary["edit_locality"]) / 2
    if "edit_reversal_success" in summary.columns:
        summary["edit_score"] = (summary["edit_success"] + summary["edit_locality"] + summary["edit_reversal_success"]) / 3
    if "edit_resource_success" in summary.columns:
        summary["edit_score"] = (
            summary["edit_success"]
            + summary["edit_resource_success"]
            + summary["edit_locality"]
            + summary.get("edit_resource_locality", summary["edit_locality"])
            + summary.get("edit_reversal_success", 0.0)
        ) / 5
    barplot(summary, "edit_score", "figures/edit_score.png", "Edit score")
    if "edit_resource_success" in summary.columns:
        barplot(summary, "edit_resource_success", "figures/edit_resource_success.png", "Edit resource success")
    barplot(summary, "internalization_score", "figures/internalization_score.png", "Internalization score")
    if "gated_internalization_score" in summary.columns:
        barplot(summary, "gated_internalization_score", "figures/gated_internalization_score.png", "Gated internalization score")


if __name__ == "__main__":
    main()
