from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bar_chart(rows: list[dict[str, Any]], key: str, title: str, output: str) -> None:
    Path("figures").mkdir(exist_ok=True)
    labels = [row.get("model", "") for row in rows]
    values = [float(row.get(key, 0.0)) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#3f7f93")
    plt.ylim(0, max(1.0, max(values, default=0.0) * 1.1))
    plt.ylabel(key)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b6_risk_constrained_summary.csv")
    args = parser.parse_args()
    rows = load_rows(args.summary)
    bar_chart(rows, "b6_risk_constrained_score", "B6 Risk-Constrained Scores", "figures/b6_risk_constrained_scores.png")
    bar_chart(rows, "actionability_mask_accuracy", "B6 Actionability Accuracy", "figures/b6_actionability_accuracy.png")
    bar_chart(rows, "unsafe_action_rejection_rate", "B6 Unsafe Rejection", "figures/b6_unsafe_rejection.png")
    bar_chart(rows, "indirect_intervention_accuracy", "B6 Indirect Intervention", "figures/b6_indirect_intervention.png")
    bar_chart(rows, "cost_sensitive_planning_accuracy", "B6 Cost-Sensitive Planning", "figures/b6_cost_sensitive_planning.png")
    bar_chart(rows, "risk_aware_feedback_revision_accuracy", "B6 Risk Feedback Revision", "figures/b6_risk_feedback_revision.png")
    bar_chart(rows, "gain_over_risk_blind", "B6 Baseline Comparison", "figures/b6_baseline_comparison.png")


if __name__ == "__main__":
    main()
