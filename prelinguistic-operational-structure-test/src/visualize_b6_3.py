from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b6_3_structural_necessity_summary.csv")
    args = parser.parse_args()
    rows = read_csv(Path(args.summary))
    drops = {}
    for row in rows:
        ablation = row["ablation_name"]
        if ablation in {"reference", "baseline", "reference_b62"}:
            continue
        drops.setdefault(ablation, []).append(float(row["observed_performance_drop"]))
    labels = sorted(drops)
    values = [sum(drops[label]) / len(drops[label]) for label in labels]
    Path("figures").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), values, color="#355c7d")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel("Mean performance drop")
    plt.title("B6.3 Structural Necessity Ablation Drops")
    plt.tight_layout()
    plt.savefig("figures/b6_3_structural_necessity_plot.png", dpi=160)
    plt.close()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
