from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .data import canonical_context, generate_dataset, make_counterfactual_pair, resource_rule
from .extraction import ExtractedRelationTable, extract_table_from_model, predict_resource_from_model, table_alignment
from .features import dataset_tensors
from .intervention import subspace_intervention_drop
from .metrics import gated_internalization_score
from .models import EditPressureModel, ExplicitTableOracle
from .probes import (
    collect_hidden_states,
    nuisance_probe_accuracy,
    relation_probe_accuracy,
    train_nuisance_probe,
    train_relation_probe,
)


def evaluate_accuracy(model, dataset: list[dict[str, str]]) -> float:
    if isinstance(model, ExplicitTableOracle):
        hits = [1.0 if model.predict_resource(record) == record["resource"] else 0.0 for record in dataset]
        return sum(hits) / len(hits)
    if isinstance(model, EditPressureModel):
        hits = [1.0 if predict_resource_from_model(model, record)[0] == record["resource"] else 0.0 for record in dataset]
        return sum(hits) / len(hits)
    x, y = dataset_tensors(dataset)
    with torch.no_grad():
        pred = torch.argmax(model(x), dim=-1)
        return float((pred == y).float().mean().item())


def evaluate_shortcut_rejection(model, attack_dataset: list[dict[str, str]]) -> float:
    return evaluate_accuracy(model, attack_dataset)


def evaluate_reversal_adaptation(model, reversal_dataset: list[dict[str, str]]) -> float:
    if isinstance(model, ExplicitTableOracle):
        oracle = ExplicitTableOracle("reversal")
        return evaluate_accuracy(oracle, reversal_dataset)
    table = extract_table_from_model(model)
    for texture, wet, outcome in [("A", "dry", "poison"), ("A", "wet", "food"), ("B", "dry", "food"), ("B", "wet", "food")]:
        table.edit_rule({"texture": texture, "wet": wet}, outcome)
    return evaluate_table_accuracy(table, reversal_dataset)


def evaluate_counterfactual_consistency(model, records: list[dict[str, str]]) -> float:
    hits = []
    for record in records[:200]:
        left, right = make_counterfactual_pair(record, "nuisance_change")
        l_pred, _ = predict_resource_from_model(model, left)
        r_pred, _ = predict_resource_from_model(model, right)
        hits.append(1.0 if l_pred == r_pred == record["resource"] else 0.0)
        _, rel = make_counterfactual_pair(record, "relation_change")
        rel_pred, _ = predict_resource_from_model(model, rel)
        hits.append(1.0 if rel_pred == rel["resource"] else 0.0)
    return sum(hits) / len(hits)


def evaluate_table_accuracy(table: ExtractedRelationTable, dataset: list[dict[str, str]]) -> float:
    hits = [1.0 if table.infer_resource(record) == record["resource"] else 0.0 for record in dataset]
    return sum(hits) / len(hits)


def evaluate_table_edit(table: ExtractedRelationTable) -> tuple[float, float]:
    before = {key: rule.outcome for key, rule in table.rules.items()}
    ok = table.edit_rule({"texture": "A", "wet": "dry"}, "poison")
    after = {key: rule.outcome for key, rule in table.rules.items()}
    target_changed = ok and after[("A", "dry")] == "poison"
    non_target_keys = [key for key in before if key != ("A", "dry")]
    locality = sum(1.0 if before[key] == after[key] else 0.0 for key in non_target_keys) / len(non_target_keys)
    return (1.0 if target_changed else 0.0), locality


def evaluate_extracted_table(table: ExtractedRelationTable, test_sets: dict[str, list[dict[str, str]]]) -> dict[str, float]:
    return {
        "table_alignment": table_alignment(table),
        "table_ood_accuracy": evaluate_table_accuracy(table, test_sets["ood"]),
        "table_spurious_attack_accuracy": evaluate_table_accuracy(table, test_sets["attack"]),
    }


def evaluate_all(model, config: dict[str, Any], seed: int, model_name: str) -> dict[str, float]:
    train = generate_dataset(int(config["n_test"]), seed + 1, "base", "train")
    ood = generate_dataset(int(config["n_test"]), seed + 2, "base", "none")
    attack = generate_dataset(int(config["n_test"]), seed + 3, "base", "attack")
    reversal = generate_dataset(int(config["n_test"]), seed + 4, "reversal", "none")
    probe_data = generate_dataset(500, seed + 5, "base", "none")

    table = extract_table_from_model(model)
    table_metrics = evaluate_extracted_table(table, {"ood": ood, "attack": attack})
    table_edit_success, edit_locality = evaluate_table_edit(table)

    hidden = collect_hidden_states(model, probe_data)
    relation_probe = train_relation_probe(hidden, probe_data)
    nuisance_probe = train_nuisance_probe(hidden, probe_data)
    relation_acc = relation_probe_accuracy(relation_probe, hidden, probe_data)
    nuisance_acc = nuisance_probe_accuracy(nuisance_probe, hidden, probe_data)
    relation_drop = subspace_intervention_drop(model, ood, relation_probe, "relation")
    nuisance_drop = subspace_intervention_drop(model, ood, nuisance_probe, "nuisance")

    # Explicit oracle has no neural subspace; keep it as an upper bound for table behavior.
    if isinstance(model, ExplicitTableOracle):
        relation_acc = 1.0
        nuisance_acc = 0.0
        relation_drop = 1.0
        nuisance_drop = 0.0

    metrics = {
        "train_accuracy": evaluate_accuracy(model, train),
        "ood_accuracy": evaluate_accuracy(model, ood),
        "spurious_attack_accuracy": evaluate_accuracy(model, attack),
        "shortcut_rejection_accuracy": evaluate_shortcut_rejection(model, attack),
        "reversal_adaptation_accuracy": evaluate_reversal_adaptation(model, reversal),
        "counterfactual_consistency": evaluate_counterfactual_consistency(model, ood),
        "probe_relation_accuracy": relation_acc,
        "probe_nuisance_accuracy": nuisance_acc,
        "probe_selectivity": relation_acc - 0.25,
        "relation_subspace_drop": relation_drop,
        "nuisance_subspace_drop": nuisance_drop,
        **table_metrics,
        "table_edit_success": table_edit_success,
        "edit_locality": edit_locality,
    }
    metrics["gated_internalization_score"] = gated_internalization_score(metrics, config["gates"])
    output = Path("results/extracted_tables")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{model_name}_seed{seed}.json").write_text(json.dumps(table.to_rows(), indent=2), encoding="utf-8")
    return metrics


def evaluate_train_accuracy(model, dataset):
    return evaluate_accuracy(model, dataset)


def evaluate_ood(model, dataset):
    return evaluate_accuracy(model, dataset)
