from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b51_closed_loop_audit_summary.csv")
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_metric(rows, "b51_closed_loop_audit_score", "B5.1 audit score", "figures/b51_closed_loop_audit_scores.png")
    plot_metric(rows, "cross_model_exact_plan_match_rate", "Plan overlap", "figures/b51_plan_overlap.png")
    plot_metric(rows, "private_trace_closed_loop_usage_rate", "Policy provenance", "figures/b51_policy_provenance.png")
    plot_metric(rows, "decision_diversity_score", "Decision diversity", "figures/b51_decision_diversity.png")
    plot_metric(rows, "trace_update_specificity", "Trace update specificity", "figures/b51_trace_update_specificity.png")
    plot_metric(rows, "feedback_revision_specificity", "Feedback revision specificity", "figures/b51_feedback_revision_specificity.png")
    plot_pair(rows, "model_gain_over_inspect_always", "model_gain_over_intervene_immediately", "Baseline sanity", "figures/b51_baseline_sanity.png")
    plot_metric(rows, "value_leakage_count", "Value leakage audit", "figures/b51_value_leakage_audit.png")
    plot_metric(rows, "planning_budget_stress_retention", "Planning budget stress", "figures/b51_planning_budget_stress.png")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def labels(rows: list[dict[str, str]]) -> list[str]:
    return [row["model"].replace("_model", "") for row in rows]


def plot_metric(rows: list[dict[str, str]], key: str, title: str, output: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(labels(rows), [float(row.get(key, 0.0)) for row in rows], color="#4d6f8c")
    plt.title(title)
    plt.ylabel(key)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_pair(rows: list[dict[str, str]], left: str, right: str, title: str, output: str) -> None:
    xs = range(len(rows))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar([x - width / 2 for x in xs], [float(row.get(left, 0.0)) for row in rows], width=width, label=left, color="#345995")
    plt.bar([x + width / 2 for x in xs], [float(row.get(right, 0.0)) for row in rows], width=width, label=right, color="#d65f5f")
    plt.title(title)
    plt.xticks(list(xs), labels(rows), rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


if __name__ == "__main__":
    main()
