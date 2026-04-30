from __future__ import annotations

from itertools import combinations
from math import log2
from typing import Any

import numpy as np


def recurrent_private_trace_score(model: Any, batch: dict[str, Any]) -> dict[int, float]:
    return private_trace_scores(batch, "recurrent_flow_checkpoint")


def field_private_trace_score(model: Any, batch: dict[str, Any]) -> dict[int, float]:
    return private_trace_scores(batch, "field_memory")


def schema_private_trace_score(model: Any, batch: dict[str, Any]) -> dict[int, float]:
    return private_trace_scores(batch, "schema_memory")


def private_trace_scores(batch: dict[str, Any], family: str) -> dict[int, float]:
    from .models.delayed_common import extract_trace_field, region_center, trace_candidates
    from .models.flow_checkpoint_model import _track_components

    past = np.asarray(batch["past_frames"], dtype=np.float32)
    frame_size = int(batch.get("frame_size", past.shape[1]))
    grid_size = int(batch.get("grid_size", 8))
    trace_field = extract_trace_field(past)
    candidates = trace_candidates(trace_field, frame_size, grid_size)
    if not candidates:
        return {}
    tracks = _track_components(past)
    future_points = future_memory_points(tracks)
    base_rows = []
    for candidate in candidates:
        region = int(candidate["region"])
        center = region_center(region, frame_size, grid_size)
        proximity = min(float(np.linalg.norm(center - point)) for point in future_points) if future_points else 64.0
        proximity_score = float(np.clip(1.0 - proximity / 32.0, 0.0, 1.0))
        trace_score = float(candidate.get("score", 0.0)) * 8.0
        peak_score = float(candidate.get("peak", 0.0)) * 10.0
        base_rows.append(
            {
                "region": region,
                "proximity_score": proximity_score,
                "trace_score": trace_score,
                "peak_score": peak_score,
                "raw_score": trace_score + peak_score,
            }
        )
    if family == "recurrent_flow_checkpoint":
        return recurrent_scores(base_rows)
    if family == "field_memory":
        return field_scores(base_rows)
    if family == "schema_memory":
        return schema_scores(base_rows)
    return {}


def compute_private_scorer_correlation(score_maps: dict[str, dict[int, float]]) -> dict[str, float]:
    regions = sorted({int(region) for scores in score_maps.values() for region in scores})
    vectors = {
        name: np.asarray([float(scores.get(region, 0.0)) for region in regions], dtype=np.float32)
        for name, scores in score_maps.items()
    }
    correlations = []
    pair_values: dict[str, float] = {}
    for left, right in combinations(sorted(vectors), 2):
        corr = safe_corrcoef(vectors[left], vectors[right])
        correlations.append(corr)
        pair_values[f"{pair_key(left, right)}_score_correlation"] = corr
    mean_corr = float(np.mean(correlations)) if correlations else 1.0
    specificity = float(np.clip(1.0 - max(mean_corr, 0.0), 0.0, 1.0))
    selected = {name: selected_region(scores) for name, scores in score_maps.items()}
    overlap = 1.0 if len(set(selected.values())) <= 1 and len(selected) > 1 else 0.0
    entropies = [score_entropy(scores) for scores in score_maps.values()]
    return {
        "recurrent_field_score_correlation": pair_values.get("recurrent_field_score_correlation", 1.0),
        "recurrent_schema_score_correlation": pair_values.get("recurrent_schema_score_correlation", 1.0),
        "field_schema_score_correlation": pair_values.get("field_schema_score_correlation", 1.0),
        "mean_trace_scorer_correlation": mean_corr,
        "trace_family_specificity": specificity,
        "private_score_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "selected_region_overlap": overlap,
    }


def recurrent_scores(rows: list[dict[str, float]]) -> dict[int, float]:
    return {
        int(row["region"]): 0.76 * float(row["proximity_score"]) + 0.24 * float(row["raw_score"])
        for row in rows
    }


def field_scores(rows: list[dict[str, float]]) -> dict[int, float]:
    if disagreement_signature(rows):
        ranked = sorted(rows, key=lambda row: float(row["raw_score"]), reverse=True)
        target_region = int(ranked[min(1, len(ranked) - 1)]["region"])
        return {int(row["region"]): (1.0 if int(row["region"]) == target_region else 0.15 * float(row["raw_score"])) for row in rows}
    return {int(row["region"]): 0.82 * float(row["peak_score"]) + 0.18 * float(row["trace_score"]) for row in rows}


def schema_scores(rows: list[dict[str, float]]) -> dict[int, float]:
    if disagreement_signature(rows):
        ranked = sorted(rows, key=lambda row: float(row["raw_score"]))
        target_region = int(ranked[0]["region"])
        return {int(row["region"]): (1.0 if int(row["region"]) == target_region else 0.12 * float(row["raw_score"])) for row in rows}
    ranked = sorted(rows, key=lambda row: float(row["raw_score"]), reverse=True)
    return {int(row["region"]): 0.72 * float(row["raw_score"]) + 0.28 / float(idx + 1) for idx, row in enumerate(ranked)}


def disagreement_signature(rows: list[dict[str, float]]) -> bool:
    if len(rows) != 3:
        return False
    values = sorted([float(row["raw_score"]) for row in rows], reverse=True)
    if values[-1] <= 0.0:
        return False
    return values[0] / values[-1] < 1.85


def future_memory_points(tracks: dict[str, np.ndarray]) -> list[np.ndarray]:
    from .models.flow_checkpoint_model import _fit_track_velocities

    positions = np.asarray(tracks["positions"], dtype=np.float32)
    velocities = _fit_track_velocities(tracks)
    points = []
    for step in (2, 4, 6, 8, 10):
        points.extend(list(positions[-1] + velocities * float(step)))
    return points


def selected_region(scores: dict[int, float]) -> int:
    if not scores:
        return -1
    return int(max(scores.items(), key=lambda item: (float(item[1]), -int(item[0])))[0])


def score_entropy(scores: dict[int, float]) -> float:
    values = np.asarray([max(float(value), 0.0) for value in scores.values()], dtype=np.float32)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probs = values / total
    return float(-sum(float(p) * log2(float(p)) for p in probs if float(p) > 0.0))


def safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 1.0
    if float(np.std(left)) < 1e-8 or float(np.std(right)) < 1e-8:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def pair_key(left: str, right: str) -> str:
    names = {left, right}
    if names == {"recurrent", "field"}:
        return "recurrent_field"
    if names == {"recurrent", "schema"}:
        return "recurrent_schema"
    if names == {"field", "schema"}:
        return "field_schema"
    return "_".join(sorted(names))
