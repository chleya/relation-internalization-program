from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .models.delayed_common import extract_trace_field, region_center, trace_candidates
from .models.flow_checkpoint_model import _fit_track_velocities, _track_components


TRACE_FAMILY_BY_MODEL = {
    "recurrent_flow_checkpoint_model": "recurrent_flow_checkpoint",
    "recurrent_flow_checkpoint_no_shared_selector": "recurrent_flow_checkpoint",
    "field_memory_model": "field_memory",
    "field_memory_no_shared_selector": "field_memory",
    "schema_memory_model": "schema_memory",
    "schema_memory_no_shared_selector": "schema_memory",
}


def score_recurrent_trace_regions(model: Any, batch: dict[str, Any]) -> dict[int, float] | dict[str, bool]:
    if trace_family_for_model(model) not in {"recurrent_flow_checkpoint", ""}:
        return {"applicable": False}
    return private_trace_scores(batch, "recurrent_flow_checkpoint")


def score_field_trace_regions(model: Any, batch: dict[str, Any]) -> dict[int, float] | dict[str, bool]:
    if trace_family_for_model(model) not in {"field_memory", ""}:
        return {"applicable": False}
    return private_trace_scores(batch, "field_memory")


def score_schema_trace_regions(model: Any, batch: dict[str, Any]) -> dict[int, float] | dict[str, bool]:
    if trace_family_for_model(model) not in {"schema_memory", ""}:
        return {"applicable": False}
    return private_trace_scores(batch, "schema_memory")


def score_private_trace_regions(model: Any, batch: dict[str, Any]) -> dict[int, float] | dict[str, bool]:
    family = trace_family_for_model(model)
    if family not in {"recurrent_flow_checkpoint", "field_memory", "schema_memory"}:
        return {"applicable": False}
    return private_trace_scores(batch, family)


def private_trace_scores(batch: dict[str, Any], family: str) -> dict[int, float]:
    past = np.asarray(batch["past_frames"], dtype=np.float32)
    frame_size = int(batch.get("frame_size", past.shape[1]))
    grid_size = int(batch.get("grid_size", 8))
    trace_field = extract_trace_field(past)
    candidates = trace_candidates(trace_field, frame_size, grid_size)
    if not candidates:
        return {}
    tracks = _track_components(past)
    positions = np.asarray(tracks["positions"], dtype=np.float32)
    velocities = _fit_track_velocities(tracks)
    future_points = []
    for step in (2, 4, 6, 8, 10):
        future_points.extend(list(positions[-1] + velocities * float(step)))

    scores: dict[int, float] = {}
    ordered = sorted(candidates, key=lambda item: int(item["region"]))
    for slot_idx, candidate in enumerate(ordered):
        region = int(candidate["region"])
        center = region_center(region, frame_size, grid_size)
        proximity = min(float(np.linalg.norm(center - point)) for point in future_points) if future_points else 64.0
        proximity_score = float(np.clip(1.0 - proximity / 32.0, 0.0, 1.0))
        trace_score = float(candidate.get("score", 0.0)) * 8.0
        peak_score = float(candidate.get("peak", 0.0)) * 10.0
        slot_score = 1.0 / float(slot_idx + 1)
        if family == "recurrent_flow_checkpoint":
            score = 0.78 * proximity_score + 0.22 * trace_score
        elif family == "field_memory":
            score = 0.82 * peak_score + 0.18 * trace_score
        else:
            score = 0.62 * slot_score + 0.24 * trace_score + 0.14 * proximity_score
        scores[region] = float(score)
    return scores


def compute_trace_scorer_correlation(scorer_outputs: dict[str, dict[int, float] | dict[str, bool]]) -> dict[str, float]:
    vectors: dict[str, np.ndarray] = {}
    regions = sorted(
        {
            int(region)
            for scores in scorer_outputs.values()
            if is_score_dict(scores)
            for region in scores
        }
    )
    for name, scores in scorer_outputs.items():
        if not is_score_dict(scores) or not regions:
            continue
        vectors[name] = np.asarray([float(scores.get(region, 0.0)) for region in regions], dtype=np.float32)

    pair_values: dict[str, float] = {}
    correlations = []
    for left, right in combinations(sorted(vectors), 2):
        corr = safe_corrcoef(vectors[left], vectors[right])
        key = pair_key(left, right)
        pair_values[f"{key}_score_correlation"] = corr
        correlations.append(corr)
    mean_corr = float(np.mean(correlations)) if correlations else 1.0
    return {
        "recurrent_field_score_correlation": pair_values.get("recurrent_field_score_correlation", 1.0),
        "recurrent_schema_score_correlation": pair_values.get("recurrent_schema_score_correlation", 1.0),
        "field_schema_score_correlation": pair_values.get("field_schema_score_correlation", 1.0),
        "mean_trace_scorer_correlation": mean_corr,
        "trace_family_specificity": float(max(0.0, 1.0 - mean_corr)),
    }


def trace_family_for_model(model: Any) -> str:
    family = str(getattr(model, "structural_family", ""))
    if family in {"recurrent_flow_checkpoint", "field_memory", "schema_memory"}:
        return family
    return TRACE_FAMILY_BY_MODEL.get(str(getattr(model, "name", "")), "")


def is_score_dict(value: Any) -> bool:
    return isinstance(value, dict) and not bool(value.get("applicable") is False)


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
