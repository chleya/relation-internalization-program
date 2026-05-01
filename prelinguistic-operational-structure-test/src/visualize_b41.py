from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b41_intervention_audit_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "b41_intervention_audit_score", "B4.1 intervention audit score", Path("figures/b41_intervention_audit_scores.png"))
    plot_bars(rows, ["cross_model_exact_action_match_rate", "exact_all_model_same_action_rate"], "Action overlap", Path("figures/b41_action_overlap.png"))
    plot_bars(rows, ["fixed_action_type_rate", "action_type_entropy"], "Action type distribution", Path("figures/b41_action_type_distribution.png"))
    plot_bars(rows, ["shared_action_policy_usage_rate", "private_trace_action_score_usage_rate", "fallback_usage_rate", "oracle_value_usage_rate"], "Action policy provenance", Path("figures/b41_action_policy_provenance.png"))
    plot_bars(rows, ["mean_action_scorer_correlation", "action_scorer_specificity"], "Action scorer correlation", Path("figures/b41_action_scorer_correlation.png"))
    plot_bars(rows, ["shared_action_policy_ablation_drop", "private_action_retention_after_shared_ablation"], "Shared policy ablation", Path("figures/b41_shared_policy_ablation.png"))
    plot_bars(rows, ["wrong_action_penalty", "wrong_region_penalty", "wrong_action_wrong_region_penalty"], "Wrong action stress", Path("figures/b41_wrong_action_stress.png"))
    plot_bars(rows, ["random_intervention_score", "saliency_intervention_score", "short_horizon_intervention_score", "inspect_only_score", "oracle_intervention_score"], "Baseline sanity", Path("figures/b41_baseline_sanity.png"))
    plot_bars(rows, ["private_trace_ablation_drop", "matched_non_trace_ablation_drop", "action_type_shift_after_trace_ablation", "region_shift_after_trace_ablation"], "Trace ablation specificity", Path("figures/b41_trace_ablation_specificity.png"))


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
    plt.ylim(0, max(1.05, max(values) * 1.10 if values else 1.05))
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()

