from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b4_intervention_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "b4_intervention_score", "B4 intervention score", Path("figures/b4_intervention_scores.png"))
    plot_metric(rows, "action_type_accuracy", "Action type accuracy", Path("figures/b4_action_type_accuracy.png"))
    plot_metric(rows, "wrong_region_penalty_sensitivity", "Wrong-region penalty", Path("figures/b4_wrong_region_penalty.png"))
    plot_metric(rows, "trace_ablation_intervention_drop", "Trace ablation effects", Path("figures/b4_trace_ablation_effects.png"))
    plot_bars(
        rows,
        ["recurrent_intervention_accuracy", "field_intervention_accuracy", "schema_intervention_accuracy"],
        "Family intervention targets",
        Path("figures/b4_family_intervention_targets.png"),
    )
    plot_bars(
        rows,
        ["random_intervention_score", "saliency_intervention_score", "short_horizon_intervention_score", "inspect_only_score", "oracle_intervention_score"],
        "Baseline comparison",
        Path("figures/b4_baseline_comparison.png"),
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

