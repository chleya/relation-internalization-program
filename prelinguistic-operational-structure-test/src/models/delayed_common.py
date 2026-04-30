from __future__ import annotations

from typing import Any

import numpy as np

from ..features import connected_components, patch_mask, point_to_region_id, region_id_to_slice, region_logits_from_map
from ..model_io import assert_clean_model_batch
from .base import logits_from_region
from .flow_checkpoint_model import _fit_track_velocities, _rollout_tracks, _track_components, _track_motion_energy


TRACE_THRESHOLD = 0.035
TRACE_MAX_INTENSITY = 0.19


def delayed_trace_forward(batch: dict[str, Any], family: str) -> dict[str, Any]:
    assert_clean_model_batch(batch)
    past = np.asarray(batch["past_frames"], dtype=np.float32)
    frame_size = int(batch["frame_size"])
    grid_size = int(batch["grid_size"])
    horizon = int(batch["future_horizon"])
    tracks = _track_components(past)
    if _track_motion_energy(tracks) < 0.20:
        future = np.repeat(past[-1][None, ...], horizon, axis=0)
        empty = np.zeros((frame_size, frame_size), dtype=np.float32)
        return {
            "future_frames": future,
            "identity_logits": None,
            "event_logits": empty,
            "inspection_logits": np.zeros(grid_size * grid_size, dtype=np.float32),
            "structure": {"applicable": False},
        }

    trace_field = extract_trace_field(past)
    candidates = trace_candidates(trace_field, frame_size, grid_size)
    selected_region = select_delayed_region(candidates, tracks, frame_size, grid_size, family)
    if selected_region is None:
        midpoint = tracks["positions"][-1].mean(axis=0)
        selected_region = int(point_to_region_id(midpoint, frame_size, grid_size))
        confidence = 0.20
    else:
        confidence = float(max(candidate["score"] for candidate in candidates if int(candidate["region"]) == int(selected_region)))

    force_vector = estimate_delayed_force(tracks)
    checkpoint = {
        "region": int(selected_region),
        "kind": "delayed_trace_checkpoint",
        "force_region": int(selected_region),
        "force_vector": force_vector,
        "collision_region": None,
        "occluder_region": None,
    }
    future = _rollout_tracks(tracks, checkpoint, horizon, frame_size, past[-1])
    structure = build_delayed_structure(
        family=family,
        trace_field=trace_field,
        candidates=candidates,
        selected_region=int(selected_region),
        confidence=confidence,
        frame_size=frame_size,
        grid_size=grid_size,
        tracks=tracks,
    )
    return {
        "future_frames": future,
        "identity_logits": np.eye(2, dtype=np.float32),
        "event_logits": structure.get("event_boundary_map", structure.get("inspection_value_field")),
        "inspection_logits": structure["inspection_logits"],
        "structure": structure,
    }


def delayed_trace_intervention(
    batch: dict[str, Any],
    intervention: dict[str, Any],
    family: str,
    causal_types: set[str],
    control_type: str = "non_trace_control",
) -> dict[str, Any]:
    output = delayed_trace_forward(batch, family)
    if not output.get("structure", {}).get("applicable", False):
        return {"applicable": False}
    kind = str(intervention.get("type", ""))
    if kind not in causal_types and kind != control_type:
        return {"applicable": False}
    past = np.asarray(batch["past_frames"], dtype=np.float32)
    frame_size = int(batch["frame_size"])
    horizon = int(batch["future_horizon"])
    tracks = _track_components(past)
    selected_region = int(np.argmax(output["inspection_logits"]))
    if kind == control_type:
        future = output["future_frames"].copy()
        control_region = far_region(selected_region, int(batch["grid_size"]))
        mask = patch_mask(frame_size, control_region)[None, :, :, None]
        future = future * (1.0 - 0.03 * mask)
        return {
            "applicable": True,
            "future_frames": future,
            "base_future_frames": output["future_frames"],
            "target": "non_trace_control",
            "target_region": control_region,
        }

    altered = {
        "region": selected_region,
        "kind": "trace_removed",
        "force_region": None,
        "force_vector": np.zeros(2, dtype=np.float32),
        "collision_region": None,
        "occluder_region": None,
    }
    future = _rollout_tracks(tracks, altered, horizon, frame_size, past[-1])
    return {
        "applicable": True,
        "future_frames": future,
        "base_future_frames": output["future_frames"],
        "target": "delayed_trace",
        "target_region": selected_region,
        "intervention_type": kind,
    }


def extract_trace_field(past: np.ndarray) -> np.ndarray:
    intensity = np.asarray(past, dtype=np.float32).mean(axis=-1)
    max_intensity = np.asarray(past, dtype=np.float32).max(axis=-1)
    faint = (intensity > TRACE_THRESHOLD) & (max_intensity < TRACE_MAX_INTENSITY)
    trace = np.where(faint, intensity, 0.0)
    return trace.mean(axis=0).astype(np.float32)


def trace_candidates(trace_field: np.ndarray, frame_size: int, grid_size: int) -> list[dict[str, float]]:
    components = connected_components(trace_field > TRACE_THRESHOLD, min_size=2)
    candidates: list[dict[str, float]] = []
    for component in components:
        yx = component.mean(axis=0)
        region = int(point_to_region_id((float(yx[1]), float(yx[0])), frame_size, grid_size))
        ys, xs = region_id_to_slice(region, frame_size, grid_size)
        strength = float(trace_field[ys, xs].mean())
        peak = float(trace_field[ys, xs].max())
        candidates.append({"region": float(region), "strength": strength, "peak": peak, "score": strength + peak})
    if candidates:
        return dedupe_candidates(candidates)

    logits = region_logits_from_map(trace_field, grid_size)
    region = int(np.argmax(logits))
    if float(logits[region]) <= 0.0:
        return []
    return [{"region": float(region), "strength": float(logits[region]), "peak": float(logits[region]), "score": float(logits[region])}]


def select_delayed_region(
    candidates: list[dict[str, float]],
    tracks: dict[str, np.ndarray],
    frame_size: int,
    grid_size: int,
    family: str,
) -> int | None:
    if not candidates:
        return None
    positions = np.asarray(tracks["positions"], dtype=np.float32)
    velocities = _fit_track_velocities(tracks)
    future_points = []
    for step in (2, 3, 4, 5, 6, 7, 8):
        future_points.extend(list(positions[-1] + velocities * float(step)))
    scored = []
    for candidate in candidates:
        region = int(candidate["region"])
        center = region_center(region, frame_size, grid_size)
        proximity = min(float(np.linalg.norm(center - point)) for point in future_points) if future_points else 64.0
        proximity_score = float(np.clip(1.0 - proximity / 32.0, 0.0, 1.0))
        trace_score = float(candidate["score"]) * 8.0
        if family == "recurrent_flow_checkpoint":
            score = 0.62 * proximity_score + 0.38 * trace_score
        elif family == "field_memory":
            score = 0.35 * proximity_score + 0.65 * trace_score
        else:
            score = 0.48 * proximity_score + 0.52 * trace_score
        scored.append((score, region))
    scored.sort(reverse=True)
    return int(scored[0][1])


def build_delayed_structure(
    family: str,
    trace_field: np.ndarray,
    candidates: list[dict[str, float]],
    selected_region: int,
    confidence: float,
    frame_size: int,
    grid_size: int,
    tracks: dict[str, np.ndarray],
) -> dict[str, Any]:
    locality = np.zeros((frame_size, frame_size), dtype=np.float32)
    ys, xs = region_id_to_slice(selected_region, frame_size, grid_size)
    locality[ys, xs] = 1.0
    logits = logits_from_region(selected_region, grid_size * grid_size, strength=6.0 + confidence)
    checkpoint_logits = region_logits_from_map(trace_field, grid_size)
    checkpoint_logits[selected_region] = max(float(checkpoint_logits[selected_region]), 6.0 + confidence)
    candidate_slots = np.asarray(
        [[float(item["region"]), float(item["strength"]), float(item["peak"]), float(item["score"])] for item in candidates[:8]],
        dtype=np.float32,
    )
    if candidate_slots.size == 0:
        candidate_slots = np.zeros((0, 4), dtype=np.float32)

    if family == "recurrent_flow_checkpoint":
        return {
            "applicable": True,
            "memory_trace": np.asarray(tracks["positions"], dtype=np.float32),
            "checkpoint_logits": checkpoint_logits.astype(np.float32),
            "delayed_checkpoint_map": locality.copy(),
            "inspection_logits": logits,
            "critical_region_logits": logits,
            "event_boundary_map": locality.copy(),
            "intervention_family": "recurrent_flow_checkpoint",
        }
    if family == "field_memory":
        return {
            "applicable": True,
            "latent_field_memory": trace_field.astype(np.float32),
            "force_trace_field": trace_field.astype(np.float32),
            "uncertainty_trace_field": np.sqrt(np.maximum(trace_field, 0.0)).astype(np.float32),
            "delayed_influence_field": locality.copy(),
            "inspection_value_field": locality.copy(),
            "inspection_logits": logits,
            "critical_region_logits": logits,
            "intervention_family": "field_memory",
        }
    return {
        "applicable": True,
        "schema_memory_slots": candidate_slots,
        "delayed_candidate_slots": candidate_slots.copy(),
        "schema_checkpoint_logits": checkpoint_logits.astype(np.float32),
        "schema_delay_logits": checkpoint_logits.astype(np.float32),
        "inspection_logits": logits,
        "critical_region_logits": logits,
        "event_boundary_map": locality.copy(),
        "intervention_family": "schema_memory",
    }


def estimate_delayed_force(tracks: dict[str, np.ndarray]) -> np.ndarray:
    velocities = _fit_track_velocities(tracks)
    mean_velocity = velocities.mean(axis=0) if len(velocities) else np.asarray([1.0, 0.0], dtype=np.float32)
    normal = np.asarray([-mean_velocity[1], mean_velocity[0]], dtype=np.float32)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        normal = np.asarray([0.0, -1.0], dtype=np.float32)
    else:
        normal = normal / norm
    return (normal * 1.35).astype(np.float32)


def dedupe_candidates(candidates: list[dict[str, float]]) -> list[dict[str, float]]:
    by_region: dict[int, dict[str, float]] = {}
    for candidate in candidates:
        region = int(candidate["region"])
        existing = by_region.get(region)
        if existing is None or float(candidate["score"]) > float(existing["score"]):
            by_region[region] = candidate
    return list(by_region.values())


def region_center(region: int, frame_size: int, grid_size: int) -> np.ndarray:
    ys, xs = region_id_to_slice(region, frame_size, grid_size)
    return np.asarray([(xs.start + xs.stop - 1) / 2.0, (ys.start + ys.stop - 1) / 2.0], dtype=np.float32)


def far_region(region: int, grid_size: int) -> int:
    gy, gx = divmod(int(region), grid_size)
    candidates = []
    for candidate in range(grid_size * grid_size):
        cy, cx = divmod(candidate, grid_size)
        candidates.append((abs(cy - gy) + abs(cx - gx), candidate))
    return max(candidates)[1]
