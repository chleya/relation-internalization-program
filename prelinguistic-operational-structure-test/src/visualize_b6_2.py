from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot(rows: list[dict[str, Any]], output: str) -> None:
    Path(output).parent.mkdir(exist_ok=True)
    policies = ["b62_policy", "mask_only", "trace_only", "state_only", "conservative_uncertainty", "risk_blind", "oracle", "random"]
    conditions = sorted({row["condition"] for row in rows})
    plt.figure(figsize=(10, 5))
    for policy in policies:
        ys = []
        for condition in conditions:
            vals = [float(row["risk_constrained_score"]) for row in rows if row["policy_name"] == policy and row["condition"] == condition]
            ys.append(sum(vals) / max(1, len(vals)))
        plt.plot(conditions, ys, marker="o", label=policy)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("risk_constrained_score")
    plt.title("B6.2 explicit stress splits")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b6_2_hardening_summary.csv")
    args = parser.parse_args()
    rows = load_rows(args.summary)
    plot(rows, "results/b6_2_hardening_plot.png")
    plot(rows, "figures/b6_2_hardening_plot.png")


if __name__ == "__main__":
    main()

