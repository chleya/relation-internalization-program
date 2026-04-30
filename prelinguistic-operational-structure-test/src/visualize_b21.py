from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


GATES = {
    "false_trace_rejection": 0.75,
    "trace_swap_sensitivity": 0.70,
    "trace_deletion_specificity_ratio": 1.50,
    "multi_source_conflict_resolution": 0.70,
    "noisy_trace_robustness": 0.70,
    "trace_length_extrapolation": 0.65,
    "trace_compression_survival": 0.65,
    "true_trace_intervention_drop": 0.20,
    "non_trace_stability": 0.70,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b21_trace_hardening_summary.csv")
    args = parser.parse_args()
    rows = read_summary(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_scores(rows, Path("figures/b21_trace_hardening_scores.png"))
    plot_attack_breakdown(rows, Path("figures/b21_trace_attack_breakdown.png"))
    plot_deletion_specificity(rows, Path("figures/b21_trace_deletion_specificity.png"))
    plot_compression_curve(rows, Path("figures/b21_trace_compression_curve.png"))


def read_summary(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for row in raw:
        converted: dict[str, float | str] = {}
        for key, value in row.items():
            converted[key] = value if key == "model" else float(value or 0.0)
        rows.append(converted)
    return rows


def plot_scores(rows: list[dict[str, float | str]], output: Path) -> None:
    labels = [short_name(str(row["model"])) for row in rows]
    values = [float(row.get("b21_trace_hardening_score", 0.0)) for row in rows]
    colors = ["#2f7d5c" if value > 0 else "#b6463a" for value in values]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("B2.1 hardening score")
    ax.set_title("B2.1 trace hardening scores")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_attack_breakdown(rows: list[dict[str, float | str]], output: Path) -> None:
    keys = [
        "false_trace_rejection",
        "trace_swap_sensitivity",
        "multi_source_conflict_resolution",
        "noisy_trace_robustness",
        "trace_length_extrapolation",
        "trace_compression_survival",
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.22
    positions = list(range(len(keys)))
    for idx, row in enumerate(rows):
        offset = (idx - (len(rows) - 1) / 2.0) * width
        values = [float(row.get(key, 0.0)) for key in keys]
        ax.bar([pos + offset for pos in positions], values, width=width, label=short_name(str(row["model"])))
    ax.scatter(positions, [GATES[key] for key in keys], color="#111111", marker="_", s=300, linewidths=2, label="gate")
    ax.set_xticks(positions)
    ax.set_xticklabels([key.replace("_", "\n") for key in keys], fontsize=8)
    ax.set_ylabel("metric value")
    ax.set_title("B2.1 attack breakdown")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_deletion_specificity(rows: list[dict[str, float | str]], output: Path) -> None:
    labels = [short_name(str(row["model"])) for row in rows]
    true_drop = [float(row.get("true_trace_intervention_drop", 0.0)) for row in rows]
    control_drop = [float(row.get("matched_non_trace_drop", 0.0)) for row in rows]
    ratio = [min(float(row.get("trace_deletion_specificity_ratio", 0.0)) / GATES["trace_deletion_specificity_ratio"], 1.0) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar([value - 0.24 for value in x], true_drop, width=0.24, label="true trace drop", color="#326db3")
    ax.bar(x, control_drop, width=0.24, label="non-trace drop", color="#b6463a")
    ax.bar([value + 0.24 for value in x], ratio, width=0.24, label="ratio normalized", color="#2f7d5c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value")
    ax.set_title("B2.1 trace deletion specificity")
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_compression_curve(rows: list[dict[str, float | str]], output: Path) -> None:
    levels = [0.75, 0.50, 0.25]
    keys = ["accuracy_at_075", "accuracy_at_050", "accuracy_at_025"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for row in rows:
        ax.plot(levels, [float(row.get(key, 0.0)) for key in keys], marker="o", label=short_name(str(row["model"])))
    ax.axhline(GATES["trace_compression_survival"], color="#111111", linestyle="--", linewidth=1)
    ax.set_xlabel("compression level")
    ax.set_ylabel("accuracy")
    ax.set_title("B2.1 trace compression curve")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def short_name(model: str) -> str:
    return model.replace("_checkpoint_model", "").replace("_model", "").replace("recurrent_flow", "recurrent")


if __name__ == "__main__":
    main()
