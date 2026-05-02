from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b6_3_1_refinement_summary.csv")
    args = parser.parse_args()
    rows = read_csv(Path(args.summary))
    focus = [row for row in rows if row["policy_name"] in {"b63_1_policy", "state_only", "mask_only", "oracle"}]
    labels = sorted({row["condition"] for row in focus})
    policies = ["b63_1_policy", "state_only", "mask_only", "oracle"]
    Path("figures").mkdir(exist_ok=True)
    plt.figure(figsize=(12, 5))
    for policy in policies:
        values = [mean([row for row in focus if row["condition"] == label and row["policy_name"] == policy], "risk_constrained_score") for label in labels]
        plt.plot(range(len(labels)), values, marker="o", label=policy)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Risk-constrained score")
    plt.title("B6.3.1 Refinement Split Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/b6_3_1_refinement_plot.png", dpi=160)
    plt.close()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


if __name__ == "__main__":
    main()

