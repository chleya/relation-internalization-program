from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .agents_temporal import make_agent
from .metrics_temporal import evaluate_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--agents",
        nargs="+",
        default=[
            "surface_temporal",
            "structural_memory_temporal",
            "instant_relation_chain",
            "delayed_relation_chain",
            "learned_delayed_links",
        ],
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
    df.to_csv("results/temporal_summary.csv", index=False)

    report = [
        "# V2 Temporal Auto Report",
        "",
        "Mean over seeds:",
        "",
        "```text",
        df.groupby("agent").mean(numeric_only=True).round(3).to_string(),
        "```",
        "",
        "Claim boundary: this is a delayed relation-chain toy, not a real geotechnical time-series model.",
    ]
    Path("reports/temporal_auto_report.md").write_text("\n".join(report), encoding="utf-8")
    Path("reports/V2_TEMPORAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(df.groupby("agent")["gated_temporal_score"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
