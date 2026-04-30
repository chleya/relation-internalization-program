from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b3_active_inspection_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "b3_active_inspection_score", "B3 active inspection scores", Path("figures/b3_active_inspection_scores.png"))
    plot_metric(rows, "delayed_information_gain", "Delayed information gain", Path("figures/b3_information_gain.png"))
    plot_metric(rows, "trace_vs_saliency_rejection", "Trace vs saliency rejection", Path("figures/b3_trace_vs_saliency_conflict.png"))
    plot_metric(rows, "trace_ablation_inspection_drop", "Trace ablation effects", Path("figures/b3_trace_ablation_effects.png"))
    plot_grouped(
        rows,
        ["random_inspection_score", "saliency_inspection_score", "short_horizon_inspection_score", "oracle_inspection_score"],
        "Baseline comparison",
        Path("figures/b3_baseline_comparison.png"),
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_metric(rows: list[dict[str, str]], metric: str, title: str, output: Path) -> None:
    names = [short_name(row.get("model", "")) for row in rows]
    values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(names, values, color="#4f6d7a")
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
    plt.figure(figsize=(10, 4))
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
