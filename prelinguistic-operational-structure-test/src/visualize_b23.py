from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b23_private_selector_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "b23_private_selector_score", "B2.3 private selector scores", Path("figures/b23_private_selector_scores.png"))
    plot_grouped(rows, ["shared_selector_usage_rate", "model_private_score_usage_rate", "fallback_usage_rate"], "Provenance", Path("figures/b23_provenance_breakdown.png"))
    plot_metric(rows, "cross_model_exact_prediction_match_rate", "Prediction overlap", Path("figures/b23_prediction_overlap.png"))
    plot_metric(rows, "disagreement_episode_divergence", "Disagreement divergence", Path("figures/b23_disagreement_divergence.png"))
    plot_grouped(rows, ["model_private_trace_drop", "shared_selector_ablation_drop"], "Source ablation", Path("figures/b23_source_ablation.png"))
    plot_grouped(rows, ["b2_delayed_score", "b21_trace_hardening_score", "trace_family_specificity"], "Regression matrix", Path("figures/b23_regression_matrix.png"))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_metric(rows: list[dict[str, str]], metric: str, title: str, output: Path) -> None:
    names = [short_name(row.get("model", "")) for row in rows]
    values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(names, values, color="#52796f")
    plt.ylim(0.0, max(1.0, max(values, default=0.0) * 1.1))
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_grouped(rows: list[dict[str, str]], metrics: list[str], title: str, output: Path) -> None:
    names = [short_name(row.get("model", "")) for row in rows]
    xs = list(range(len(names)))
    width = 0.8 / max(1, len(metrics))
    plt.figure(figsize=(9, 4))
    for idx, metric in enumerate(metrics):
        values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
        offset = (idx - (len(metrics) - 1) / 2.0) * width
        plt.bar([x + offset for x in xs], values, width=width, label=metric)
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.xticks(xs, names, rotation=20, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def short_name(name: str) -> str:
    return (
        name.replace("_checkpoint_model", "")
        .replace("_memory_model", "")
        .replace("recurrent_flow", "recurrent")
    )


if __name__ == "__main__":
    main()
