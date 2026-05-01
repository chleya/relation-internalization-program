from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b32_mechanism_inspection_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "b32_mechanism_inspection_score", "B3.2 mechanism inspection score", Path("figures/b32_mechanism_inspection_scores.png"))
    plot_bars(
        rows,
        ["recurrent_goal_accuracy", "field_goal_accuracy", "schema_goal_accuracy"],
        "Family-specific targets",
        Path("figures/b32_family_specific_targets.png"),
    )
    plot_bars(
        rows,
        ["recurrent_value_alignment", "field_value_alignment", "schema_value_alignment"],
        "Value decomposition",
        Path("figures/b32_value_decomposition.png"),
    )
    plot_metric(rows, "task_conditioned_switch_accuracy", "Goal-conditioned switching", Path("figures/b32_goal_conditioned_switching.png"))
    plot_bars(
        rows,
        ["mechanism_disagreement_rate", "cross_model_same_region_rate"],
        "Mechanism disagreement",
        Path("figures/b32_mechanism_disagreement.png"),
    )
    plot_bars(
        rows,
        ["family_specific_trace_ablation_drop", "non_target_family_stability"],
        "Family trace ablation",
        Path("figures/b32_family_ablation.png"),
    )


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
    plt.figure(figsize=(8, 4))
    plt.bar(metrics, values, color="#66a61e")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
