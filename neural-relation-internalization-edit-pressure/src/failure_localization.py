from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .data import generate_dataset
from .extraction import canonical_support
from .features import dataset_tensors
from .intervention import remove_probe_subspace
from .models import EditPressureModel
from .probes import collect_hidden_states, train_nuisance_probe, train_relation_probe
from .train import train_model


GATE_COLUMNS = [
    "ood_accuracy",
    "shortcut_rejection_accuracy",
    "reversal_adaptation_accuracy",
    "counterfactual_consistency",
    "table_alignment",
    "table_edit_success",
    "edit_locality",
    "relation_subspace_drop",
    "nuisance_subspace_drop",
]


def read_records(path: str | Path = "results/records.csv") -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gate_failure_matrix(records: list[dict[str, str]], gates: dict[str, float], model_name: str = "edit_pressure_training") -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record["model"] != model_name:
            continue
        row: dict[str, Any] = {"model": model_name, "seed": int(record["seed"])}
        row["ood_pass"] = float(record["ood_accuracy"]) >= gates["ood_accuracy"]
        row["shortcut_rejection_pass"] = float(record["shortcut_rejection_accuracy"]) >= gates["shortcut_rejection_accuracy"]
        row["reversal_pass"] = float(record["reversal_adaptation_accuracy"]) >= gates["reversal_adaptation_accuracy"]
        row["counterfactual_pass"] = float(record["counterfactual_consistency"]) >= gates["counterfactual_consistency"]
        row["table_alignment_pass"] = float(record["table_alignment"]) >= gates["table_alignment"]
        row["table_edit_pass"] = float(record["table_edit_success"]) >= gates["table_edit_success"]
        row["edit_locality_pass"] = float(record["edit_locality"]) >= gates["edit_locality"]
        row["relation_subspace_drop_pass"] = float(record["relation_subspace_drop"]) >= gates["relation_subspace_drop"]
        row["nuisance_subspace_drop_pass"] = float(record["nuisance_subspace_drop"]) <= gates["nuisance_subspace_drop_max"]
        row["all_gates_pass"] = all(value for key, value in row.items() if key.endswith("_pass"))
        rows.append(row)
    return rows


def subspace_drop_by_seed(records: list[dict[str, str]], model_name: str = "edit_pressure_training") -> list[dict[str, Any]]:
    return [
        {
            "model": record["model"],
            "seed": int(record["seed"]),
            "relation_subspace_drop": float(record["relation_subspace_drop"]),
            "nuisance_subspace_drop": float(record["nuisance_subspace_drop"]),
            "gated_internalization_score": float(record["gated_internalization_score"]),
        }
        for record in records
        if record["model"] == model_name
    ]


def representation_tensor(model: EditPressureModel, site: str, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
    support = canonical_support()
    query_hidden = model.hidden(x)
    support_state = model.encode_support(*support).expand(query_hidden.shape[0], -1)
    if site == "query_hidden":
        return query_hidden, lambda edited: model.forward_from_hidden(edited, support_state)
    if site == "support_relation_state":
        return support_state, lambda edited: model.forward_from_hidden(query_hidden, edited)
    if site == "combined_hidden":
        combined = torch.cat([query_hidden, support_state], dim=-1)
        return combined, lambda edited: model.head(edited)
    if site == "post_edit_state":
        edit_x = x[0]
        edit_y = torch.tensor(1, dtype=torch.long)
        post = model.apply_edit(support_state[:1], model.encode_edit(edit_x, edit_y)).expand(query_hidden.shape[0], -1)
        return post, lambda edited: model.forward_from_hidden(query_hidden, edited)
    raise ValueError(f"unknown site: {site}")


def site_intervention_drop(model: EditPressureModel, dataset: list[dict[str, str]], site: str, kind: str) -> float:
    x, y = dataset_tensors(dataset)
    hidden, forward = representation_tensor(model, site, x)
    probe_data = generate_dataset(500, 91_000, "base", "none")
    probe_x, _ = dataset_tensors(probe_data)
    probe_hidden, _ = representation_tensor(model, site, probe_x)
    probe = train_relation_probe(probe_hidden.detach(), probe_data) if kind == "relation" else train_nuisance_probe(probe_hidden.detach(), probe_data)
    with torch.no_grad():
        base = torch.argmax(forward(hidden), dim=-1)
        edited = torch.argmax(forward(remove_probe_subspace(hidden, probe)), dim=-1)
        base_acc = float((base == y).float().mean().item())
        edited_acc = float((edited == y).float().mean().item())
    return max(0.0, base_acc - edited_acc)


def intervention_site_analysis(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    dataset = generate_dataset(int(config["n_test"]), 92_000, "base", "none")
    for seed in seeds:
        model = train_model("edit_pressure_training", seed, config)
        assert isinstance(model, EditPressureModel)
        for site in ["query_hidden", "support_relation_state", "combined_hidden", "post_edit_state"]:
            rows.append(
                {
                    "model": "edit_pressure_training",
                    "seed": seed,
                    "site": site,
                    "relation_subspace_drop": site_intervention_drop(model, dataset, site, "relation"),
                    "nuisance_subspace_drop": site_intervention_drop(model, dataset, site, "nuisance"),
                }
            )
    return rows


def extraction_subspace_correlation(records: list[dict[str, str]], model_name: str = "edit_pressure_training") -> list[dict[str, Any]]:
    rows = []
    filtered = [record for record in records if record["model"] == model_name]
    for metric in ["table_alignment", "table_ood_accuracy", "table_spurious_attack_accuracy", "table_edit_success", "edit_locality"]:
        xs = [float(record[metric]) for record in filtered]
        ys = [float(record["relation_subspace_drop"]) for record in filtered]
        rows.append({"model": model_name, "metric": metric, "relation_subspace_drop_correlation": pearson(xs, ys)})
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0.0 or y_var == 0.0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / ((x_var * y_var) ** 0.5)


def build_failure_report(
    gate_rows: list[dict[str, Any]],
    drop_rows: list[dict[str, Any]],
    site_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
) -> str:
    failed_relation = [row for row in gate_rows if not row["relation_subspace_drop_pass"]]
    site_means = {}
    for site in sorted({row["site"] for row in site_rows}):
        site_means[site] = {
            "relation": mean(row["relation_subspace_drop"] for row in site_rows if row["site"] == site),
            "nuisance": mean(row["nuisance_subspace_drop"] for row in site_rows if row["site"] == site),
        }
    lines = [
        "# Edit-Pressure Failure Analysis",
        "",
        "## Question",
        "",
        "Why does `edit_pressure_training` have strong transfer/table/edit/locality behavior but low mean gated score?",
        "",
        "## Gate Failure Summary",
        "",
        f"- Seeds analyzed: {len(gate_rows)}.",
        f"- Seeds failing only or primarily the relation-subspace gate: {len(failed_relation)}.",
        "- The per-seed gate matrix is in `results/gate_failure_matrix.csv`.",
        "",
        "## Subspace Drop Distribution",
        "",
        "| seed | relation_subspace_drop | nuisance_subspace_drop | gated |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in drop_rows:
        lines.append(
            f"| {row['seed']} | {row['relation_subspace_drop']:.3f} | {row['nuisance_subspace_drop']:.3f} | {row['gated_internalization_score']:.3f} |"
        )
    lines += [
        "",
        "## Intervention Site Analysis",
        "",
        "| site | mean_relation_drop | mean_nuisance_drop |",
        "| --- | ---: | ---: |",
    ]
    for site, values in site_means.items():
        lines.append(f"| {site} | {values['relation']:.3f} | {values['nuisance']:.3f} |")
    lines += [
        "",
        "## Extraction/Subspace Correlation",
        "",
        "| extraction_metric | correlation_with_relation_drop |",
        "| --- | ---: |",
    ]
    for row in corr_rows:
        lines.append(f"| {row['metric']} | {row['relation_subspace_drop_correlation']:.3f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "`edit_pressure_training` should not be described as a stable success. It learns relation-usable behavior at the extracted-table level, but the causal linear relation-subspace signal is unstable across seeds under the current intervention test.",
        "",
        "This supports the narrower interpretation: extracted-table editability and causal linear subspace structure are distinct diagnostics.",
        "",
        "## Boundary",
        "",
        "This analysis does not add a new experiment, does not change the model, and does not prove or disprove neural relation understanding. It localizes the current failure mode in a toy diagnostic.",
        "",
    ]
    return "\n".join(lines)
