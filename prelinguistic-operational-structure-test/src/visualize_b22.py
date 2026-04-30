from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b22_selector_disentanglement_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "shared_selector_usage_rate", "Shared selector usage", Path("figures/b22_selector_usage.png"))
    plot_metric(rows, "cross_model_exact_prediction_match_rate", "Prediction overlap", Path("figures/b22_prediction_overlap.png"))
    plot_metric(rows, "disagreement_episode_divergence", "Disagreement divergence", Path("figures/b22_disagreement_divergence.png"))
    plot_ablation(rows, Path("figures/b22_ablation_effects.png"))
    plot_metric(rows, "mean_trace_scorer_correlation", "Trace scorer correlation", Path("figures/b22_trace_scorer_correlation.png"))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_metric(rows: list[dict[str, str]], metric: str, title: str, output: Path) -> None:
    names = [short_name(row.get("model", "")) for row in rows]
    values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(names, values, color="#4b8bbe")
    plt.ylim(0.0, max(1.0, max(values, default=0.0) * 1.1))
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_ablation(rows: list[dict[str, str]], output: Path) -> None:
    names = [short_name(row.get("model", "")) for row in rows]
    private = [float(row.get("model_private_trace_drop", 0.0) or 0.0) for row in rows]
    shared = [float(row.get("shared_selector_ablation_drop", 0.0) or 0.0) for row in rows]
    xs = list(range(len(names)))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar([x - width / 2 for x in xs], private, width=width, label="private trace")
    plt.bar([x + width / 2 for x in xs], shared, width=width, label="shared selector")
    plt.ylim(0.0, 1.05)
    plt.title("Source-specific ablation effects")
    plt.ylabel("drop")
    plt.xticks(xs, names, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def short_name(name: str) -> str:
    return (
        name.replace("_checkpoint_model", "")
        .replace("_memory_model", "")
        .replace("recurrent_flow", "recurrent")
        .replace("field", "field")
        .replace("schema", "schema")
    )


if __name__ == "__main__":
    main()
