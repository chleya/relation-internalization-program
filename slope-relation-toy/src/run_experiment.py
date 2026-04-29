from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .agents import make_agent
from .evaluate import evaluate_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["majority", "surface", "structural_memory", "learned_links", "generic_review", "relation_chain"],
    )
    args = parser.parse_args()

    rows = []
    for agent_name in args.agents:
        for seed in args.seeds:
            metrics = evaluate_agent(make_agent(agent_name), seed=seed)
            rows.append({"agent": agent_name, "seed": seed, **metrics})

    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/summary.csv", index=False)
    report = [
        "# Auto Report: Slope Relation Toy",
        "",
        "Mean over seeds:",
        "",
        "```text",
        df.groupby("agent")[
            [
                "ood_success",
                "spurious_attack_success",
                "counterfactual_accuracy",
                "edit_success",
                "relation_audit",
                "irrelevant_link_rejection",
                "noisy_observation_success",
                "review_score",
                "review_consistency",
                "gated_slope_score",
            ]
        ].mean().round(3).to_string(),
        "```",
        "",
        "Claim boundary: this is a toy relation-chain diagnostic, not a real slope safety model.",
    ]
    Path("reports/auto_report.md").write_text("\n".join(report), encoding="utf-8")
    print(df.groupby("agent")["gated_slope_score"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
