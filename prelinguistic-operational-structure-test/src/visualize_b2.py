from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


GATES = {
    "delayed_checkpoint_accuracy": 0.75,
    "multi_delay_stability": 0.70,
    "early_saliency_rejection": 0.75,
    "delay_ood_generalization": 0.70,
    "causal_trace_intervention_drop": 0.20,
    "non_trace_stability": 0.70,
    "delayed_endpoint_shift": 0.25,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b2_delayed_checkpoint_summary.csv")
    args = parser.parse_args()
    rows = read_summary(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_scores(rows, Path("figures/b2_delayed_checkpoint_scores.png"))
    plot_gate_breakdown(rows, Path("figures/b2_delay_gate_breakdown.png"))
    plot_trace_effects(rows, Path("figures/b2_trace_intervention_effects.png"))


def read_summary(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for row in raw:
        converted: dict[str, float | str] = {}
        for key, value in row.items():
            if key == "model":
                converted[key] = value
            else:
                converted[key] = float(value) if value not in {"", None} else 0.0
        rows.append(converted)
    return rows


def plot_scores(rows: list[dict[str, float | str]], output: Path) -> None:
    labels = [short_name(str(row["model"])) for row in rows]
    values = [float(row.get("b2_delayed_score", 0.0)) for row in rows]
    colors = ["#2f7d5c" if value > 0 else "#b6463a" for value in values]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("B2 delayed score")
    ax.set_title("B2 delayed checkpoint substrate scores")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_gate_breakdown(rows: list[dict[str, float | str]], output: Path) -> None:
    keys = list(GATES)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.18
    positions = list(range(len(keys)))
    for idx, row in enumerate(rows):
        offset = (idx - (len(rows) - 1) / 2.0) * width
        values = [float(row.get(key, 0.0)) for key in keys]
        ax.bar([pos + offset for pos in positions], values, width=width, label=short_name(str(row["model"])))
    ax.scatter(positions, [GATES[key] for key in keys], color="#111111", marker="_", s=300, linewidths=2, label="gate")
    ax.set_xticks(positions)
    ax.set_xticklabels([key.replace("_", "\n") for key in keys], fontsize=8)
    ax.set_ylabel("metric value")
    ax.set_title("B2 gate breakdown")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_trace_effects(rows: list[dict[str, float | str]], output: Path) -> None:
    labels = [short_name(str(row["model"])) for row in rows]
    causal = [float(row.get("causal_trace_intervention_drop", 0.0)) for row in rows]
    stability = [float(row.get("non_trace_stability", 0.0)) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar([value - 0.18 for value in x], causal, width=0.36, label="causal trace drop", color="#326db3")
    ax.bar([value + 0.18 for value in x], stability, width=0.36, label="non-trace stability", color="#d09b2c")
    ax.axhline(GATES["causal_trace_intervention_drop"], color="#326db3", linewidth=1, linestyle="--")
    ax.axhline(GATES["non_trace_stability"], color="#d09b2c", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value")
    ax.set_title("B2 trace intervention effects")
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def short_name(model: str) -> str:
    return (
        model.replace("_checkpoint_model", "")
        .replace("_model", "")
        .replace("recurrent_flow", "recurrent")
        .replace("flow_checkpoint", "flow")
    )


if __name__ == "__main__":
    main()
