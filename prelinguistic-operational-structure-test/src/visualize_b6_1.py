from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


POLICIES = [
    "hardening_policy",
    "risk_blind_policy",
    "mask_only_policy",
    "always_abstain_policy",
    "oracle_risk_policy",
    "random_policy",
]


def load_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stress_intensity(row: dict[str, Any]) -> float:
    return max(
        float(row.get("mask_noise_rate", 0.0)),
        float(row.get("mask_missing_rate", 0.0)),
        float(row.get("delay_steps", 0.0)) / 5.0,
        float(row.get("inspect_cost", 0.0)) / 0.2 if float(row.get("inspect_cost", 0.0)) else 0.0,
        {"clean_correlated": 0.0, "visible_flag": 0.0, "hidden_state": 0.5, "hard_uncorrelated": 0.7, "hard_flipped": 1.0}.get(row.get("spurious_mode", ""), 0.0),
    )


def plot_summary(rows: list[dict[str, Any]], output: str) -> None:
    Path(output).parent.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 5))
    for policy in POLICIES:
        policy_rows = [row for row in rows if row.get("policy_name") == policy]
        by_x: dict[float, list[float]] = {}
        for row in policy_rows:
            x = round(stress_intensity(row), 2)
            by_x.setdefault(x, []).append(float(row.get("risk_constrained_score", 0.0)))
        xs = sorted(by_x)
        ys = [sum(by_x[x]) / len(by_x[x]) for x in xs]
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=1.8, label=policy.replace("_policy", ""))
    plt.xlabel("stress intensity")
    plt.ylabel("risk_constrained_score")
    plt.ylim(0, 1.05)
    plt.title("B6.1 Reviewer-Hardening Degradation")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b6_1_hardening_summary.csv")
    args = parser.parse_args()
    rows = load_rows(args.summary)
    plot_summary(rows, "results/b6_1_hardening_plot.png")
    plot_summary(rows, "figures/b6_1_hardening_plot.png")


if __name__ == "__main__":
    main()

