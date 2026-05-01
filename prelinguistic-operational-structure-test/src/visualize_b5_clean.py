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


def grouped_clean_vs_original(rows: list[dict[str, Any]], output: str) -> None:
    Path("figures").mkdir(exist_ok=True)
    labels = [row.get("model", "") for row in rows]
    clean = [float(row.get("clean_b5_closed_loop_score", 0.0)) for row in rows]
    original = [float(row.get("original_b5_closed_loop_score", 0.0)) for row in rows]
    x = list(range(len(labels)))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar([value - width / 2 for value in x], original, width=width, label="original", color="#777777")
    plt.bar([value + width / 2 for value in x], clean, width=width, label="clean", color="#3f7f93")
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("B5 Clean vs Original")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def leakage_chart(rows: list[dict[str, Any]], output: str) -> None:
    Path("figures").mkdir(exist_ok=True)
    labels = [row.get("model", "") for row in rows]
    values = [float(row.get("value_leakage_count", 0.0)) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#9b4d3a")
    plt.ylabel("value_leakage_count")
    plt.title("B5.1 Clean Leakage Audit")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b5_clean_closed_loop_summary.csv")
    parser.add_argument("--audit", default="results/b51_clean_closed_loop_audit_summary.csv")
    args = parser.parse_args()
    summary = load_rows(args.summary)
    audit = load_rows(args.audit)
    bar_chart(summary, "clean_b5_closed_loop_score", "B5 Clean Closed-Loop Scores", "figures/b5_clean_closed_loop_scores.png")
    grouped_clean_vs_original(summary, "figures/b5_clean_vs_original.png")
    bar_chart(audit, "b51_clean_closed_loop_audit_score", "B5.1 Clean Audit Scores", "figures/b51_clean_audit_scores.png")
    leakage_chart(audit, "figures/b51_clean_leakage_audit.png")


if __name__ == "__main__":
    main()
