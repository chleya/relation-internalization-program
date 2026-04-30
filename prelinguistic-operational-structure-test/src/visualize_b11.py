from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


B11_GATES = {
    "dynamic_decoy_rejection": 0.80,
    "delayed_checkpoint_accuracy": 0.75,
    "competing_checkpoint_choice": 0.75,
    "relocation_ood_stability": 0.75,
    "causal_over_visual_deletion_ratio": 1.50,
    "anti_prior_survival": 0.70,
    "causal_endpoint_shift": 0.25,
}

DISPLAY_KEYS = [
    "dynamic_decoy_rejection",
    "delayed_checkpoint_accuracy",
    "competing_checkpoint_choice",
    "relocation_ood_stability",
    "causal_over_visual_deletion_ratio",
    "anti_prior_survival",
    "causal_endpoint_shift",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b11_flow_checkpoint_hardening_summary.csv")
    args = parser.parse_args()
    metrics = read_summary(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_hardening(metrics, Path("figures/b11_flow_checkpoint_hardening.png"))
    plot_breakdown(metrics, Path("figures/b11_attack_breakdown.png"))


def read_summary(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {key: float(value) for key, value in rows[0].items() if value not in {"", None}}


def plot_hardening(metrics: dict[str, float], output: Path) -> None:
    labels = [key.replace("_", "\n") for key in DISPLAY_KEYS]
    values = [float(metrics.get(key, 0.0)) for key in DISPLAY_KEYS]
    gates = [B11_GATES[key] for key in DISPLAY_KEYS]
    colors = ["#2f7d5c" if value >= gate else "#b6463a" for value, gate in zip(values, gates)]
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = list(range(len(DISPLAY_KEYS)))
    ax.bar(positions, values, color=colors)
    ax.scatter(positions, gates, color="#111111", marker="_", s=360, linewidths=2, label="gate")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("metric value")
    ax.set_title("B1.1 flow-checkpoint hardening")
    ax.set_ylim(0, max(1.65, max(values + gates) + 0.1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_breakdown(metrics: dict[str, float], output: Path) -> None:
    labels = ["decoy", "delayed", "competing", "relocation", "deletion", "anti-prior"]
    values = [
        float(metrics.get("dynamic_decoy_rejection", 0.0)),
        float(metrics.get("delayed_checkpoint_accuracy", 0.0)),
        float(metrics.get("competing_checkpoint_choice", 0.0)),
        float(metrics.get("relocation_ood_stability", 0.0)),
        min(float(metrics.get("causal_over_visual_deletion_ratio", 0.0)) / B11_GATES["causal_over_visual_deletion_ratio"], 1.0),
        float(metrics.get("anti_prior_survival", 0.0)),
    ]
    gates = [0.80, 0.75, 0.75, 0.75, 1.0, 0.70]
    colors = ["#2f7d5c" if value >= gate else "#b6463a" for value, gate in zip(values, gates)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = list(range(len(labels)))
    ax.bar(positions, values, color=colors)
    ax.axhline(1.0, color="#111111", linewidth=1, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("normalized pass signal")
    ax.set_title("B1.1 attack breakdown")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
