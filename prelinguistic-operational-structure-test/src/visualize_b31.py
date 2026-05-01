from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b31_inspection_audit_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "b31_inspection_audit_score", "B3.1 inspection audit score", Path("figures/b31_inspection_audit_scores.png"))
    plot_metric(rows, "cross_model_inspect_region_match_rate", "Inspect region overlap", Path("figures/b31_inspect_region_overlap.png"))
    plot_bars(
        rows,
        ["shared_inspection_policy_usage_rate", "private_trace_inspection_score_usage_rate", "fallback_usage_rate"],
        "Policy provenance",
        Path("figures/b31_policy_provenance.png"),
    )
    plot_bars(
        rows,
        ["mean_inspection_scorer_correlation", "inspection_scorer_specificity"],
        "Inspection scorer correlation",
        Path("figures/b31_scorer_correlation.png"),
    )
    plot_metric(rows, "disagreement_inspection_divergence", "Disagreement inspection divergence", Path("figures/b31_disagreement_inspection.png"))
    plot_metric(rows, "shared_policy_ablation_drop", "Shared policy ablation drop", Path("figures/b31_shared_policy_ablation.png"))
    plot_bars(
        rows,
        ["random_inspection_score", "saliency_inspection_score", "short_horizon_inspection_score", "oracle_inspection_score"],
        "Baseline sanity",
        Path("figures/b31_baseline_sanity.png"),
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_metric(rows: list[dict[str, str]], metric: str, title: str, path: Path) -> None:
    labels = [str(row.get("seed", idx)) for idx, row in enumerate(rows)]
    values = [float(row.get(metric, 0.0)) for row in rows]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#4477aa")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_bars(rows: list[dict[str, str]], metrics: list[str], title: str, path: Path) -> None:
    if not rows:
        return
    values = [float(rows[0].get(metric, 0.0)) for metric in metrics]
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
