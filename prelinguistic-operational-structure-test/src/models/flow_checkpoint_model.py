from __future__ import annotations

from typing import Any

import numpy as np

from ..features import connected_components, draw_disk, extract_blob_centers, patch_mask, point_to_region_id, rect_to_region_id, region_id_to_slice
from ..model_io import assert_clean_model_batch
from .base import BasePLOSModel, logits_from_region


class FlowCheckpointModel(BasePLOSModel):
    name = "flow_checkpoint_model"
    structural_family = "flow_checkpoint"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert_clean_model_batch(batch)
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        if float(past.max()) < 0.18:
            future = np.repeat(past[-1][None, ...], int(batch["future_horizon"]), axis=0)
            empty_map = np.zeros((int(batch["frame_size"]), int(batch["frame_size"])), dtype=np.float32)
            return {
                "future_frames": future,
                "identity_logits": None,
                "event_logits": empty_map,
                "inspection_logits": np.zeros(int(batch["grid_size"]) ** 2, dtype=np.float32),
                "structure": {"applicable": False},
            }
        tracks = _track_components(past)
        if _track_motion_energy(tracks) < 0.20:
            future = np.repeat(past[-1][None, ...], int(batch["future_horizon"]), axis=0)
            empty_map = np.zeros((int(batch["frame_size"]), int(batch["frame_size"])), dtype=np.float32)
            return {
                "future_frames": future,
                "identity_logits": None,
                "event_logits": empty_map,
                "inspection_logits": np.zeros(int(batch["grid_size"]) ** 2, dtype=np.float32),
                "structure": {"applicable": False},
            }
        checkpoint = _select_checkpoint(past, tracks, int(batch["frame_size"]), int(batch["grid_size"]), int(batch["future_horizon"]))
        future = _rollout_tracks(tracks, checkpoint, int(batch["future_horizon"]), int(batch["frame_size"]), past[-1])
        structure = _structure_from_checkpoint(checkpoint, int(batch["frame_size"]), int(batch["grid_size"]))
        return {
            "future_frames": future,
            "identity_logits": np.eye(2, dtype=np.float32),
            "event_logits": structure["event_boundary_map"],
            "inspection_logits": structure["critical_region_logits"],
            "structure": structure,
        }

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        output = self.forward(batch)
        kind = intervention.get("type")
        if kind not in {"event_latent_perturbation", "relation_edge_ablation", "inspection_map_shuffle", "inspection_topk_zero", "field_patch_mask", "critical_field_zero"}:
            return {"applicable": False}
        if not output.get("structure", {}).get("applicable", False):
            return {"applicable": False}
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        tracks = _track_components(past)
        checkpoint = _select_checkpoint(past, tracks, int(batch["frame_size"]), int(batch["grid_size"]), int(batch["future_horizon"]))
        future = output["future_frames"].copy()
        region = int(np.argmax(output["inspection_logits"]))
        mask = patch_mask(future.shape[1], region)[None, :, :, None]
        if kind in {"event_latent_perturbation", "relation_edge_ablation"}:
            altered = dict(checkpoint)
            if checkpoint.get("kind") == "force_anomaly":
                altered["force_region"] = None
                altered["force_vector"] = np.zeros(2, dtype=np.float32)
            elif checkpoint.get("kind") == "predicted_collision":
                altered["disable_collision"] = True
            else:
                altered["force_region"] = None
                altered["disable_collision"] = True
            future = _rollout_tracks(tracks, altered, int(batch["future_horizon"]), int(batch["frame_size"]), past[-1])
            target = kind
        elif kind == "inspection_map_shuffle":
            future = future * 0.95
            target = "inspection"
        else:
            future = future * (1.0 - 0.60 * mask)
            target = "field"
        return {"applicable": True, "future_frames": future, "base_future_frames": output["future_frames"], "target": target, "target_region": region}


def _track_components(past: np.ndarray, n_tracks: int = 2) -> dict[str, np.ndarray]:
    centers_by_t = [extract_blob_centers(frame) for frame in past]
    tracks = np.zeros((len(past), n_tracks, 2), dtype=np.float32)
    visible = np.zeros((len(past), n_tracks), dtype=bool)

    first = next((centers for centers in centers_by_t if len(centers) > 0), np.zeros((0, 2), dtype=np.float32))
    if len(first) >= n_tracks:
        init = first[np.argsort(first[:, 0])[:n_tracks]]
    elif len(first) == 1:
        init = np.vstack([first[0], first[0] + np.asarray([28.0, 20.0], dtype=np.float32)])
    else:
        init = np.asarray([[12.0, 18.0], [52.0, 46.0]], dtype=np.float32)
    tracks[0] = init
    visible[0, : min(len(first), n_tracks)] = True
    velocity = np.zeros((n_tracks, 2), dtype=np.float32)

    for t in range(1, len(past)):
        predicted = tracks[t - 1] + velocity
        tracks[t] = predicted
        centers = centers_by_t[t].copy()
        used: set[int] = set()
        for idx in range(n_tracks):
            if len(centers) == 0:
                continue
            distances = np.linalg.norm(centers - predicted[idx], axis=1)
            order = np.argsort(distances)
            chosen = None
            for candidate in order:
                if int(candidate) not in used:
                    chosen = int(candidate)
                    break
            if chosen is None or distances[chosen] > 14.0:
                continue
            used.add(chosen)
            previous = tracks[t - 1, idx].copy()
            tracks[t, idx] = centers[chosen]
            velocity[idx] = tracks[t, idx] - previous
            visible[t, idx] = True

    if len(past) >= 2:
        velocities = tracks[1:] - tracks[:-1]
    else:
        velocities = np.zeros((1, n_tracks, 2), dtype=np.float32)
    return {"positions": tracks, "velocities": velocities, "visible": visible}


def _track_motion_energy(tracks: dict[str, np.ndarray]) -> float:
    velocities = np.asarray(tracks["velocities"], dtype=np.float32)
    visible = np.asarray(tracks["visible"], dtype=bool)
    if len(velocities) == 0:
        return 0.0
    valid = visible[1:] & visible[:-1]
    if not valid.any():
        return 0.0
    speed = np.linalg.norm(velocities, axis=-1)
    return float(speed[valid].mean())


def _select_checkpoint(past: np.ndarray, tracks: dict[str, np.ndarray], frame_size: int, grid_size: int, horizon: int) -> dict[str, Any]:
    positions = tracks["positions"]
    velocities = tracks["velocities"]
    visible = tracks["visible"]
    accel = velocities[1:] - velocities[:-1] if len(velocities) >= 2 else np.zeros((0, 2, 2), dtype=np.float32)
    accel_norm = np.linalg.norm(accel, axis=-1) if len(accel) else np.zeros((0, 2), dtype=np.float32)
    occluder = _detect_occluder_rect(past[-1])
    force_region = None
    force_vector = np.zeros(2, dtype=np.float32)
    force_strength = 0.0
    force_time = 999
    if accel_norm.size:
        valid = np.zeros_like(accel_norm, dtype=bool)
        for time_idx in range(accel_norm.shape[0]):
            for track_idx in range(accel_norm.shape[1]):
                valid[time_idx, track_idx] = bool(visible[time_idx : time_idx + 3, track_idx].all())
        valid_accel = np.where(valid, accel_norm, 0.0)
        flat = int(np.argmax(valid_accel))
        time_idx, track_idx = np.unravel_index(flat, accel_norm.shape)
        force_time = int(time_idx)
        force_strength = float(valid_accel[time_idx, track_idx])
        if force_strength > 0.65 and force_time <= 2:
            point = positions[min(time_idx, len(positions) - 1), track_idx]
            force_region = int(point_to_region_id(point, frame_size, grid_size))
            force_vector = accel[time_idx, track_idx].astype(np.float32)

    collision_region, collision_score = _predict_collision_region(positions[-1], velocities[-1] if len(velocities) else np.zeros((2, 2)), frame_size, grid_size, horizon)
    occluder_region = rect_to_region_id(occluder, frame_size, grid_size) if occluder is not None else None

    if force_region is not None:
        region = force_region
        kind = "force_anomaly"
        score = min(1.0, force_strength * 2.5)
    elif collision_region is not None and collision_score > 0.55:
        if occluder_region is not None and _region_distance(collision_region, occluder_region, grid_size) <= 2:
            region = occluder_region
            kind = "occlusion_checkpoint"
            score = 0.85
        else:
            region = collision_region
            kind = "predicted_collision"
            score = collision_score
    elif occluder_region is not None:
        region = occluder_region
        kind = "occlusion_checkpoint"
        score = 0.85
    elif collision_region is not None and collision_score > 0.25:
        region = collision_region
        kind = "predicted_collision"
        score = collision_score
    else:
        midpoint = positions[-1].mean(axis=0)
        region = int(point_to_region_id(midpoint, frame_size, grid_size))
        kind = "motion_midpoint"
        score = 0.35

    return {
        "region": int(region),
        "kind": kind,
        "score": float(score),
        "force_vector": force_vector,
        "force_region": force_region,
        "collision_region": collision_region,
        "occluder_region": occluder_region,
    }


def _region_distance(a: int, b: int, grid_size: int) -> int:
    ay, ax = divmod(int(a), grid_size)
    by, bx = divmod(int(b), grid_size)
    return abs(ay - by) + abs(ax - bx)


def _predict_collision_region(positions: np.ndarray, velocities: np.ndarray, frame_size: int, grid_size: int, horizon: int) -> tuple[int | None, float]:
    if len(positions) < 2 or len(velocities) < 2:
        return None, 0.0
    relative_pos = positions[0] - positions[1]
    relative_vel = velocities[0] - velocities[1]
    denom = float(np.dot(relative_vel, relative_vel))
    if denom < 1e-6:
        return None, 0.0
    time = float(np.clip(-np.dot(relative_pos, relative_vel) / denom, 0.0, float(horizon)))
    p0 = positions[0] + velocities[0] * time
    p1 = positions[1] + velocities[1] * time
    distance = float(np.linalg.norm(p0 - p1))
    score = float(np.clip(1.0 - distance / 18.0, 0.0, 1.0))
    if score <= 0.0:
        return None, 0.0
    return int(point_to_region_id((p0 + p1) / 2.0, frame_size, grid_size)), score


def _detect_occluder_rect(frame: np.ndarray) -> tuple[float, float, float, float] | None:
    intensity = np.asarray(frame, dtype=np.float32).mean(axis=-1)
    mask = (intensity > 0.04) & (intensity < 0.16)
    components = connected_components(mask, min_size=32)
    if not components:
        return None
    comp = max(components, key=len)
    ys = comp[:, 0]
    xs = comp[:, 1]
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    area_fraction = ((x1 - x0) * (y1 - y0)) / float(frame.shape[0] * frame.shape[1])
    if area_fraction > 0.35:
        return None
    return (x0, y0, x1, y1)


def _rollout_tracks(tracks: dict[str, np.ndarray], checkpoint: dict[str, Any], horizon: int, frame_size: int, template: np.ndarray) -> np.ndarray:
    positions = tracks["positions"][-1].copy()
    velocities = _fit_track_velocities(tracks)
    force_region = checkpoint.get("force_region")
    force_vector = np.asarray(checkpoint.get("force_vector", np.zeros(2, dtype=np.float32)), dtype=np.float32)
    frames = []
    colors = [np.asarray([0.95, 0.25, 0.25], dtype=np.float32), np.asarray([0.25, 0.65, 1.0], dtype=np.float32)]
    for _ in range(horizon):
        if force_region is not None:
            for idx, point in enumerate(positions):
                if int(point_to_region_id(point, frame_size, 8)) == int(force_region):
                    velocities[idx] = velocities[idx] + force_vector
        if len(positions) >= 2 and not checkpoint.get("disable_collision", False):
            relative_pos = positions[0] - positions[1]
            relative_vel = velocities[0] - velocities[1]
            if np.linalg.norm(relative_pos) < 14.0 and float(np.dot(relative_pos, relative_vel)) < 0.0:
                velocities[0], velocities[1] = velocities[1].copy(), velocities[0].copy()
        positions = positions + velocities
        for axis in (0, 1):
            hit = (positions[:, axis] < 4.0) | (positions[:, axis] > frame_size - 5.0)
            velocities[hit, axis] *= -1
            positions[:, axis] = np.clip(positions[:, axis], 4.0, frame_size - 5.0)
        frame = np.zeros_like(template)
        for idx, point in enumerate(positions[:2]):
            draw_disk(frame, point, 4.0, colors[idx % len(colors)])
        frames.append(frame)
    return np.asarray(frames, dtype=np.float32)


def _fit_track_velocities(tracks: dict[str, np.ndarray]) -> np.ndarray:
    positions = tracks["positions"]
    visible = tracks["visible"]
    n_tracks = positions.shape[1]
    fitted = np.zeros((n_tracks, 2), dtype=np.float32)
    times = np.arange(len(positions), dtype=np.float32)
    fallback = tracks["velocities"][-1] if len(tracks["velocities"]) else fitted
    for idx in range(n_tracks):
        mask = visible[:, idx]
        if int(mask.sum()) >= 3:
            t = times[mask]
            p = positions[mask, idx]
            t_centered = t - float(t.mean())
            denom = float((t_centered**2).sum())
            if denom > 1e-6:
                fitted[idx] = (t_centered[:, None] * (p - p.mean(axis=0))).sum(axis=0) / denom
            else:
                fitted[idx] = fallback[idx]
        else:
            fitted[idx] = fallback[idx]
    return fitted.astype(np.float32)


def _structure_from_checkpoint(checkpoint: dict[str, Any], frame_size: int, grid_size: int) -> dict[str, Any]:
    region = int(checkpoint["region"])
    locality = np.zeros((frame_size, frame_size), dtype=np.float32)
    ys, xs = region_id_to_slice(region, frame_size, grid_size)
    locality[ys, xs] = 1.0
    event_map = np.zeros_like(locality)
    for candidate in {region, checkpoint.get("force_region"), checkpoint.get("collision_region"), checkpoint.get("occluder_region")}:
        if candidate is not None:
            _paint_region_neighborhood(event_map, int(candidate), frame_size, grid_size)
    logits = logits_from_region(region, grid_size * grid_size, strength=6.0)
    return {
        "applicable": True,
        "checkpoint_region": region,
        "checkpoint_kind": checkpoint["kind"],
        "relation_locality_map": locality,
        "event_boundary_map": event_map,
        "inspection_value_field": locality,
        "critical_region_logits": logits,
        "intervention_family": "flow_checkpoint",
    }


def _paint_region_neighborhood(target: np.ndarray, region: int, frame_size: int, grid_size: int) -> None:
    gy, gx = divmod(int(region), grid_size)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny = gy + dy
            nx = gx + dx
            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                ys, xs = region_id_to_slice(ny * grid_size + nx, frame_size, grid_size)
                target[ys, xs] = 1.0
