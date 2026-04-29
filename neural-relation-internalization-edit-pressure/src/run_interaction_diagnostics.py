from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .interaction_diagnostics import evaluate_interaction_diagnostics_for_model, write_csv


DEFAULT_MODELS = ["pure_prediction", "counterfactual_training", "edit_pressure_training"]


def rows_by_model(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["model"] == model]


def mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    return mean(float(row[metric]) for row in rows) if rows else 0.0


def aggregate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [key for key in rows[0].keys() if key not in {"model", "seed"}]
    summary = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = rows_by_model(rows, model)
        summary.append({"model": model, "seed": "mean", **{metric: mean_metric(model_rows, metric) for metric in metrics}})
    return summary


def compact_ablation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model and row["applicable"]]
        for site in sorted({row["site"] for row in model_rows}):
            site_rows = [row for row in model_rows if row["site"] == site]
            compact.append(
                {
                    "model": model,
                    "site": site,
                    "max_drop": max(float(row["accuracy_drop"]) for row in site_rows),
                    "mean_drop": mean(float(row["accuracy_drop"]) for row in site_rows),
                }
            )
    return compact


def build_report(summary: list[dict[str, Any]], nonlinear: list[dict[str, Any]], shuffle: list[dict[str, Any]], swap: list[dict[str, Any]], binding: list[dict[str, Any]], ablation: list[dict[str, Any]]) -> str:
    aggregated = aggregate_summary(summary)
    ablation_compact = compact_ablation_summary(ablation)
    edit_rows = [row for row in aggregated if row["model"] == "edit_pressure_training"]
    observed = edit_rows[0] if edit_rows else {}
    lines = [
        "# V1.2 Nonlinear / Interaction Causal Check",
        "",
        "## 1. Motivation",
        "",
        "V1.1 found that `edit_pressure_training` has extracted-table editability but unstable `relation_subspace_drop`. V1.2 asks whether the relation structure is absent, or whether it is encoded in a nonlinear, interaction-mediated, episode-conditioned form that linear probe-subspace intervention can miss.",
        "",
        "## 2. Diagnostics Added",
        "",
        "- nonlinear probes",
        "- support-state shuffle",
        "- edit-state swap",
        "- query-support binding",
        "- multi-site ablation matrix",
        "",
        "## 3. Results",
        "",
        "### Interaction Summary",
        "",
        "| model | nonlinear gain | support shuffle | edit swap | binding acc | support ablation | interaction evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregated:
        lines.append(
            f"| {row['model']} | {row['nonlinear_relation_gain']:.3f} | {row['support_shuffle_drop']:.3f} | "
            f"{row['edit_state_swap_success']:.3f} | {row['support_conditioned_accuracy']:.3f} | "
            f"{row['max_support_site_ablation_drop']:.3f} | {row['interaction_evidence_score']:.3f} |"
        )
    if observed:
        if observed["support_shuffle_drop"] < 0.05 and observed["support_conditioned_accuracy"] < 0.75:
            interpretation = "The observed edit-pressure pattern is weak support-query interaction evidence: support shuffle has little effect and same-query support binding is not reliable."
        elif observed["support_shuffle_drop"] >= 0.15 and observed["support_conditioned_accuracy"] >= 0.75:
            interpretation = "The observed edit-pressure pattern supports interaction-mediated relation use: support perturbations affect behavior and same-query predictions follow support regime."
        else:
            interpretation = "The observed edit-pressure pattern is mixed and should be treated as seed- or mechanism-specific."
        if observed["edit_state_swap_success"] >= 0.75:
            interpretation += " Edit-state swap is strong, so the edit pathway has behavioral effect even when support-conditioned regime use remains weak."
        lines += [
            "",
            "Observed diagnostic pattern:",
            "",
            interpretation,
        ]
    lines += [
        "",
        "### Nonlinear Probe Comparison",
        "",
        "| model | linear relation | MLP relation | tree relation | linear nuisance | MLP nuisance | tree nuisance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_summary(nonlinear):
        lines.append(
            f"| {row['model']} | {row['linear_relation_acc']:.3f} | {row['mlp_relation_acc']:.3f} | "
            f"{row['tree_relation_acc']:.3f} | {row['linear_nuisance_acc']:.3f} | "
            f"{row['mlp_nuisance_acc']:.3f} | {row['tree_nuisance_acc']:.3f} |"
        )
    lines += [
        "",
        "### Support Shuffle",
        "",
        "| model | normal | shuffled | drop |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in aggregate_summary(shuffle):
        lines.append(f"| {row['model']} | {row['normal_accuracy']:.3f} | {row['shuffled_support_accuracy']:.3f} | {row['support_shuffle_drop']:.3f} |")
    lines += [
        "",
        "### Edit State Swap",
        "",
        "| model | normal edit acc | swapped effect | swapped locality | success |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_summary(swap):
        lines.append(
            f"| {row['model']} | {row['normal_edit_accuracy']:.3f} | {row['swapped_edit_effect_rate']:.3f} | "
            f"{row['swapped_edit_locality']:.3f} | {row['edit_state_swap_success']:.3f} |"
        )
    lines += [
        "",
        "### Binding Test",
        "",
        "| model | support-conditioned acc | binding sensitivity | invariance failure |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in aggregate_summary(binding):
        lines.append(
            f"| {row['model']} | {row['support_conditioned_accuracy']:.3f} | "
            f"{row['binding_sensitivity']:.3f} | {row['support_invariance_failure_rate']:.3f} |"
        )
    lines += [
        "",
        "### Ablation Matrix Summary",
        "",
        "| model | site | max drop | mean drop |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in ablation_compact:
        lines.append(f"| {row['model']} | {row['site']} | {row['max_drop']:.3f} | {row['mean_drop']:.3f} |")
    lines += [
        "",
        "## 4. Interpretation",
        "",
        "A. Strong nonlinear/interaction evidence would mean edit-pressure relation use may exist outside a single linear subspace.",
        "",
        "B. Weak nonlinear/interaction evidence would mean edit-pressure mainly learned shallow editable patterns rather than strong support/edit-state causal use.",
        "",
        "C. Seed-specific evidence would mean the training pressure can sometimes form relation structure, but the mechanism is unstable across seeds.",
        "",
        "## 5. Relation to V1.1",
        "",
        "This does not overturn the original gated result. It explains why `edit_pressure_training` may fail linear relation-subspace diagnostics despite table-level editability. The diagnostic `interaction_evidence_score` is not a replacement for `gated_internalization_score`.",
        "",
        "## 6. Claim Boundary",
        "",
        "Supported:",
        "- diagnostic localization of edit-pressure representation form in a toy setting.",
        "",
        "Unsupported:",
        "- proof of general neural relation internalization.",
        "- proof of nonlinear causal representation.",
        "- large model claims.",
        "- real-world causal discovery.",
        "",
    ]
    return "\n".join(lines)


def build_self_audit() -> str:
    return "\n".join(
        [
            "# V1.2 Self-Audit",
            "",
            "## What V1.2 improves",
            "",
            "- Tests whether edit-pressure relation structure may be nonlinear.",
            "- Tests whether model uses support-conditioned relation state.",
            "- Tests whether edit state has causal effect.",
            "- Tests whether query-support binding exists.",
            "- Extends intervention beyond linear subspace removal.",
            "",
            "## Remaining weaknesses",
            "",
            "- Toy world remains simple.",
            "- Probes are still diagnostic, not proof of understanding.",
            "- Nonlinear probes can overfit small datasets.",
            "- Support shuffle may disrupt distribution in artificial ways.",
            "- Edit-state swap requires model-specific hooks.",
            "- Interaction evidence score is not a replacement for gated internalization score.",
            "- No claim of general causal representation.",
            "",
            "## False positive risks",
            "",
            "- MLP probe may decode information that the policy does not use.",
            "- Swap tests may exploit architecture artifacts.",
            "- Ablation may cause out-of-distribution hidden states.",
            "- Support-conditioned behavior may still be shallow memorization.",
            "",
            "## Required failure checks",
            "",
            "1. MLP probe high but support shuffle low.",
            "2. Edit-state swap has no behavioral effect.",
            "3. Query-support binding fails.",
            "4. Ablation effects are only present under OOD-hidden artifacts.",
            "5. Interaction evidence is strong only for one seed.",
            "",
        ]
    )


def run(config_path: str) -> list[dict[str, Any]]:
    sweep = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base_config = yaml.safe_load(Path(sweep["base_config"]).read_text(encoding="utf-8"))
    models = [model for model in sweep["models"] if model in DEFAULT_MODELS]
    seeds = [int(seed) for seed in sweep["seeds"]]

    summary_rows: list[dict[str, Any]] = []
    nonlinear_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    for model_name in models:
        for seed in seeds:
            result = evaluate_interaction_diagnostics_for_model(model_name, seed, base_config)
            summary_rows.append(result["summary"])
            nonlinear_rows.append(result["nonlinear"])
            shuffle_rows.append(result["shuffle"])
            swap_rows.append(result["swap"])
            binding_rows.append(result["binding"])
            for row in result["ablation"]:
                ablation_rows.append({**row, "model": model_name})
            print(f"{model_name} seed={seed} interaction_evidence={result['summary']['interaction_evidence_score']:.3f}")

    write_csv("results/nonlinear_probe_summary.csv", nonlinear_rows)
    write_csv("results/support_shuffle_results.csv", shuffle_rows)
    write_csv("results/edit_state_swap_results.csv", swap_rows)
    write_csv("results/query_support_binding_results.csv", binding_rows)
    write_csv("results/ablation_matrix.csv", ablation_rows)
    write_csv("results/interaction_diagnostics_summary.csv", summary_rows)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/V1_2_INTERACTION_DIAGNOSTICS.md").write_text(build_report(summary_rows, nonlinear_rows, shuffle_rows, swap_rows, binding_rows, ablation_rows), encoding="utf-8")
    Path("reports/V1_2_SELF_AUDIT.md").write_text(build_self_audit(), encoding="utf-8")
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
