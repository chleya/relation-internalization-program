from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b21a_degeneracy_audit_summary.csv")
    args = parser.parse_args()
    summary = read_single_row(Path(args.summary))
    Path("figures").mkdir(exist_ok=True)
    plot_score_degeneracy(summary, Path("figures/b21a_score_degeneracy.png"))
    plot_per_attack(Path("results/b21a_per_attack_breakdown.csv"), Path("figures/b21a_per_attack_breakdown.png"))
    plot_region_distribution(Path("results/b21a_predicted_region_distribution.csv"), Path("figures/b21a_predicted_region_distribution.png"))
    plot_baselines(Path("results/b21a_baseline_comparison.csv"), Path("figures/b21a_baseline_comparison.png"))
    plot_ablation(summary, Path("figures/b21a_ablation_effects.png"))


def read_single_row(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: float(value or 0.0) for key, value in rows[0].items()} if rows else {}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_score_degeneracy(summary: dict[str, float], output: Path) -> None:
    keys = [
        "score_degeneracy_detected",
        "all_attacks_identical_flag",
        "exact_same_score_all_models_all_seeds",
        "cross_model_exact_prediction_match_rate",
        "gt_region_match_rate",
        "b21a_degeneracy_audit_score",
    ]
    values = [summary.get(key, 0.0) for key in keys]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([key.replace("_", "\n") for key in keys], values, color="#326db3")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value")
    ax.set_title("B2.1a score degeneracy audit")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_per_attack(path: Path, output: Path) -> None:
    rows = read_rows(path)
    attacks = sorted({row["attack"] for row in rows})
    models = sorted({row["model"] for row in rows})
    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = 0.22
    positions = list(range(len(attacks)))
    for idx, model in enumerate(models):
        values = [next((float(row["metric_value"]) for row in rows if row["model"] == model and row["attack"] == attack), 0.0) for attack in attacks]
        ax.bar([pos + (idx - 1) * width for pos in positions], values, width=width, label=short_name(model))
    ax.set_xticks(positions)
    ax.set_xticklabels([attack.replace("_", "\n") for attack in attacks], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("metric value")
    ax.set_title("B2.1a per-attack breakdown")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_region_distribution(path: Path, output: Path) -> None:
    rows = read_rows(path)
    models = sorted({row["model"] for row in rows})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        counts: dict[int, float] = {}
        for row in model_rows:
            region = int(float(row["region_id"]))
            counts[region] = counts.get(region, 0.0) + float(row["count"])
        total = sum(counts.values()) or 1.0
        xs = sorted(counts)
        ax.plot(xs, [counts[x] / total for x in xs], marker="o", linewidth=1, label=short_name(model))
    ax.set_xlabel("region id")
    ax.set_ylabel("frequency")
    ax.set_title("B2.1a predicted region distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_baselines(path: Path, output: Path) -> None:
    rows = read_rows(path)
    score_rows = [row for row in rows if row["metric_name"] in {"random_b21_score", "oracle_b21_score"}]
    labels = [row["baseline"].replace("_", "\n") for row in score_rows]
    values = [float(row["metric_value"]) for row in score_rows]
    colors = ["#b6463a" if "random" in label else "#2f7d5c" for label in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("B2.1 score")
    ax.set_title("B2.1a baseline comparison")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_ablation(summary: dict[str, float], output: Path) -> None:
    keys = ["trace_family_ablation_drop", "no_trace_ablation_drop"]
    values = [summary.get(key, 0.0) for key in keys]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar([key.replace("_", "\n") for key in keys], values, color=["#326db3", "#d09b2c"])
    ax.axhline(0.20, color="#326db3", linestyle="--", linewidth=1)
    ax.axhline(0.30, color="#d09b2c", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("drop")
    ax.set_title("B2.1a ablation effects")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def short_name(model: str) -> str:
    return model.replace("_checkpoint_model", "").replace("_model", "").replace("recurrent_flow", "recurrent")


if __name__ == "__main__":
    main()
