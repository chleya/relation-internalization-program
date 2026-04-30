from __future__ import annotations

from statistics import mean
from typing import Any

from .interventions import default_interventions
from .metrics_behavior import behavior_metrics
from .metrics_intervention import intervention_metrics
from .model_io import make_model_batch


def evaluate_behavior(model: Any, dataset: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    rows = []
    for episode in dataset:
        batch = make_model_batch(episode, config)
        rows.append(behavior_metrics(model.forward(batch), episode))
    return _mean_rows(rows)


def evaluate_structure_interventions(model: Any, dataset: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    rows = []
    for episode in dataset:
        batch = make_model_batch(episode, config)
        results = {}
        for intervention in default_interventions():
            results[intervention["type"]] = model.intervene_structure(batch, intervention)
        rows.append(intervention_metrics(results))
    return _mean_rows(rows)


def evaluate_ood(model: Any, ood_datasets: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, float]:
    scores = []
    for dataset in ood_datasets.values():
        behavior = evaluate_behavior(model, dataset, config)
        scores.append(
            mean(
                [
                    behavior.get("identity_after_occlusion", 0.0),
                    behavior.get("identity_after_crossing", 0.0),
                    behavior.get("event_boundary_alignment", 0.0),
                    behavior.get("critical_region_selection_accuracy", 0.0),
                ]
            )
        )
    return {"ood_trajectory_generalization": mean(scores) if scores else 0.0}


def evaluate_all(model: Any, datasets: dict[str, Any], config: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        "behavior": evaluate_behavior(model, datasets["test"], config),
        "structure": evaluate_structure_interventions(model, datasets["test"], config),
        "ood": evaluate_ood(model, datasets["ood"], config),
    }


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: mean(float(row.get(key, 0.0)) for row in rows) for key in keys}
