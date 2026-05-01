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
    parser.add_argument("--summary", default="results/b52_adaptive_update_summary.csv")
    args = parser.parse_args()
    rows = load_rows(args.summary)
    bar_chart(rows, "b52_adaptive_update_score", "B5.2 Adaptive Update Scores", "figures/b52_adaptive_update_scores.png")
    bar_chart(rows, "inspection_content_sensitivity", "Inspection Content Sensitivity", "figures/b52_inspection_content_sensitivity.png")
    bar_chart(rows, "counterfactual_update_switch_rate", "Counterfactual Update Switch", "figures/b52_counterfactual_update_switch.png")
    bar_chart(rows, "post_update_plan_divergence", "Post-Update Plan Divergence", "figures/b52_plan_divergence.png")
    bar_chart(rows, "feedback_content_sensitivity", "Feedback Sensitivity", "figures/b52_feedback_sensitivity.png")
    bar_chart(rows, "model_gain_over_scripted_update", "Scripted Baseline Comparison", "figures/b52_scripted_baseline_comparison.png")
    bar_chart(rows, "revision_specific_ablation_drop", "Revision Ablation", "figures/b52_revision_ablation.png")


if __name__ == "__main__":
    main()
