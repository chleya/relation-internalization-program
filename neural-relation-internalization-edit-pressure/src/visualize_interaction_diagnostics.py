from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grouped_means(rows: list[dict[str, str]], metrics: list[str]) -> list[dict[str, float | str]]:
    models = sorted({row["model"] for row in rows})
    output: list[dict[str, float | str]] = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        item: dict[str, float | str] = {"model": model}
        for metric in metrics:
            item[metric] = sum(float(row[metric]) for row in model_rows) / len(model_rows)
        output.append(item)
    return output


def bar_plot(rows: list[dict[str, float | str]], metrics: list[str], output: str, ylim: tuple[float, float] = (0, 1.05)) -> None:
    models = [str(row["model"]) for row in rows]
    width = 0.8 / max(len(metrics), 1)
    x = list(range(len(models)))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, metric in enumerate(metrics):
        ax.bar([pos + i * width for pos in x], [float(row[metric]) for row in rows], width=width, label=metric)
    ax.set_xticks([pos + width * (len(metrics) - 1) / 2 for pos in x])
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(output).parent.mkdir(exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def ablation_heatmap(rows: list[dict[str, str]], output: str) -> None:
    filtered = [row for row in rows if row.get("applicable") == "True" or row.get("applicable") is True]
    if not filtered:
        return
    sites = sorted({row["site"] for row in filtered})
    models = sorted({row["model"] for row in filtered})
    matrix = []
    for site in sites:
        values = []
        for model in models:
            selected = [float(row["accuracy_drop"]) for row in filtered if row["model"] == model and row["site"] == site]
            values.append(max(selected) if selected else 0.0)
        matrix.append(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites)
    ax.set_title("max ablation accuracy drop")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    Path(output).parent.mkdir(exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/interaction_diagnostics_summary.csv")
    args = parser.parse_args()
    rows = read_rows(args.summary)
    means = grouped_means(
        rows,
        [
            "linear_relation_acc",
            "mlp_relation_acc",
            "tree_relation_acc",
            "support_shuffle_drop",
            "edit_state_swap_success",
            "support_conditioned_accuracy",
            "binding_sensitivity",
        ],
    )
    bar_plot(means, ["linear_relation_acc", "mlp_relation_acc", "tree_relation_acc"], "figures/nonlinear_probe_comparison.png")
    bar_plot(means, ["support_shuffle_drop"], "figures/support_shuffle_drop.png")
    bar_plot(means, ["edit_state_swap_success"], "figures/edit_state_swap.png")
    bar_plot(means, ["support_conditioned_accuracy", "binding_sensitivity"], "figures/query_support_binding.png")
    ablation_heatmap(read_rows("results/ablation_matrix.csv"), "figures/ablation_matrix.png")


if __name__ == "__main__":
    main()
