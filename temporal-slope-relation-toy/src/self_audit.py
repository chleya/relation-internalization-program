from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/temporal_summary.csv")
    parser.add_argument("--output", default="reports/temporal_self_audit.md")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    grouped = df.groupby("agent").mean(numeric_only=True)
    report = [
        "# Temporal Self Audit",
        "",
        "This is not a real geotechnical time-series model.",
        "It only tests delayed relation-chain internalization in a deterministic toy world.",
        "",
        "## Mean Scores",
        "",
        "```text",
        grouped.round(3).to_string(),
        "```",
        "",
        "## Known Weaknesses",
        "",
        "- Delays are predefined candidates, not unrestricted discovery.",
        "- Sequence length is short.",
        "- State variables are directly observable in several baselines.",
        "- No calibrated safety threshold is claimed.",
        "- No real monitoring data is used.",
    ]
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
