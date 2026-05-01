from __future__ import annotations

from typing import Any

import numpy as np

from .b23_private_scorers import compute_private_scorer_correlation, private_trace_scores, selected_region
from .model_io import make_model_batch


FAMILY_FOR_MODEL = {
    "recurrent_flow_checkpoint_model": "recurrent_flow_checkpoint",
    "field_memory_model": "field_memory",
    "schema_memory_model": "schema_memory",
}


def recurrent_trace_inspection_scores(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[int, float]:
    return _inspection_scores(episode, config, "recurrent_flow_checkpoint")


def field_trace_inspection_scores(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[int, float]:
    return _inspection_scores(episode, config, "field_memory")


def schema_trace_inspection_scores(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[int, float]:
    return _inspection_scores(episode, config, "schema_memory")


def model_trace_inspection_scores(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[int, float]:
    name = str(getattr(model, "name", ""))
    family = FAMILY_FOR_MODEL.get(name, str(getattr(model, "structural_family", "")))
    return _inspection_scores(episode, config, family)


def _inspection_scores(episode: dict[str, Any], config: dict[str, Any], family: str) -> dict[int, float]:
    batch = make_model_batch(episode, config)
    return private_trace_scores(batch, family)


def compute_inspection_scorer_correlation(scorer_outputs: dict[str, dict[int, float]]) -> dict[str, float]:
    trace_metrics = compute_private_scorer_correlation(scorer_outputs)
    return {
        "recurrent_field_inspection_score_correlation": trace_metrics.get("recurrent_field_score_correlation", 1.0),
        "recurrent_schema_inspection_score_correlation": trace_metrics.get("recurrent_schema_score_correlation", 1.0),
        "field_schema_inspection_score_correlation": trace_metrics.get("field_schema_score_correlation", 1.0),
        "mean_inspection_scorer_correlation": trace_metrics.get("mean_trace_scorer_correlation", 1.0),
        "inspection_scorer_specificity": trace_metrics.get("trace_family_specificity", 0.0),
        "selected_region_overlap": trace_metrics.get("selected_region_overlap", 1.0),
    }


def evaluate_inspection_scorer_specificity(
    models: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    metrics = []
    records = []
    for episode_id, episode in enumerate(episodes):
        score_maps = {
            "recurrent": recurrent_trace_inspection_scores(models.get("recurrent_flow_checkpoint_model"), episode, config),
            "field": field_trace_inspection_scores(models.get("field_memory_model"), episode, config),
            "schema": schema_trace_inspection_scores(models.get("schema_memory_model"), episode, config),
        }
        corr = compute_inspection_scorer_correlation(score_maps)
        metrics.append(corr)
        for family, scores in score_maps.items():
            region = selected_region(scores)
            records.append(
                {
                    "seed": seed,
                    "episode_id": episode_id,
                    "trace_family": family,
                    "selected_region": region,
                    "score_count": len(scores),
                    "max_score": float(max(scores.values())) if scores else 0.0,
                    **corr,
                }
            )
    return average_metric_rows(metrics), records


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "recurrent_field_inspection_score_correlation": 1.0,
            "recurrent_schema_inspection_score_correlation": 1.0,
            "field_schema_inspection_score_correlation": 1.0,
            "mean_inspection_scorer_correlation": 1.0,
            "inspection_scorer_specificity": 0.0,
            "selected_region_overlap": 1.0,
        }
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([float(row.get(key, 0.0)) for row in rows])) for key in keys}
