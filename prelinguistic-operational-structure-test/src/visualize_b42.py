from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b42_action_type_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "b42_action_type_score", "B4.2 action-type score", Path("figures/b42_action_type_scores.png"))
    plot_bars(rows, ["fixed_action_type_rate", "action_type_accuracy", "joint_region_action_accuracy"], "Action type distribution", Path("figures/b42_action_type_distribution.png"))
    plot_metric(rows, "correct_region_wrong_action_penalty", "Correct-region wrong-action penalty", Path("figures/b42_correct_region_wrong_action.png"))
    plot_bars(rows, ["family_action_mapping_accuracy", "family_action_diversity"], "Family action mapping", Path("figures/b42_family_action_mapping.png"))
    plot_metric(rows, "action_type_counterfactual_sensitivity", "Counterfactual sensitivity", Path("figures/b42_counterfactual_sensitivity.png"))
    plot_bars(rows, ["gain_over_fixed_action_baseline", "gain_over_random_action_type", "gain_over_saliency", "gain_over_short_horizon"], "Fixed-action baseline", Path("figures/b42_fixed_action_baseline.png"))
    plot_metric(rows, "action_type_ood_accuracy", "Action-type OOD", Path("figures/b42_action_type_ood.png"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_metric(rows: list[dict[str, str]], metric: str, title: str, path: Path) -> None:
    labels = [str(row.get("model", idx)) for idx, row in enumerate(rows)]
    values = [float(row.get(metric, 0.0)) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#4477aa")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_bars(rows: list[dict[str, str]], metrics: list[str], title: str, path: Path) -> None:
    if not rows:
        return
    values = [sum(float(row.get(metric, 0.0)) for row in rows) / len(rows) for metric in metrics]
    plt.figure(figsize=(9, 4))
    plt.bar(metrics, values, color="#66a61e")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()

