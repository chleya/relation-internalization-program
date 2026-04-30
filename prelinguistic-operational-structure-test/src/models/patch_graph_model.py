from __future__ import annotations

from typing import Any

import numpy as np

from ..features import patch_mask, region_id_to_slice
from ..model_io import assert_clean_model_batch
from .base import BasePLOSModel


class PatchGraphModel(BasePLOSModel):
    name = "patch_graph_model"
    structural_family = "patch_graph"

    def __init__(self) -> None:
        self.transition: np.ndarray | None = None
        self.grid_size = 8

    def fit(self, dataset: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
        self.grid_size = int((config or {}).get("env", {}).get("grid_size", 8))
        xs = []
        ys = []
        for batch in dataset:
            past = np.asarray(batch["past_frames"], dtype=np.float32)
            for idx in range(len(past) - 1):
                xs.append(_patch_activity(past[idx], self.grid_size))
                ys.append(_patch_activity(past[idx + 1], self.grid_size))
        if not xs:
            self.transition = np.eye(self.grid_size * self.grid_size, dtype=np.float32)
            return
        x = np.asarray(xs, dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32)
        ridge = 0.05 * np.eye(x.shape[1], dtype=np.float32)
        self.transition = np.linalg.solve(x.T @ x + ridge, x.T @ y).astype(np.float32)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert_clean_model_batch(batch)
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        transition = self.transition
        if transition is None:
            transition = np.eye(int(batch["grid_size"]) ** 2, dtype=np.float32)
        activity = _patch_activity(past[-1], int(batch["grid_size"]))
        prev = _patch_activity(past[-2], int(batch["grid_size"])) if len(past) > 1 else activity
        accel = activity - prev
        future = _render_patch_rollout(past[-1], activity, transition, int(batch["future_horizon"]), int(batch["grid_size"]))
        structure = _patch_structure(activity, accel, transition, int(batch["frame_size"]), int(batch["grid_size"]))
        return {
            "future_frames": future,
            "identity_logits": None,
            "event_logits": structure["event_boundary_map"],
            "inspection_logits": structure["critical_region_logits"],
            "structure": structure,
        }

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        output = self.forward(batch)
        kind = intervention.get("type")
        if kind not in {"relation_edge_ablation", "inspection_map_shuffle", "inspection_topk_zero", "field_patch_mask", "critical_field_zero"}:
            return {"applicable": False}
        future = output["future_frames"].copy()
        region = int(np.argmax(output["inspection_logits"]))
        if kind == "relation_edge_ablation":
            mask = patch_mask(future.shape[1], region)[None, :, :, None]
            future = future * (1.0 - 0.35 * mask)
            target = "relation_edge_ablation"
        elif kind == "inspection_map_shuffle":
            future = np.roll(future, shift=8, axis=2)
            target = "inspection"
        else:
            mask = patch_mask(future.shape[1], region)[None, :, :, None]
            future = future * (1.0 - 0.40 * mask)
            target = "field"
        return {"applicable": True, "future_frames": future, "base_future_frames": output["future_frames"], "target": target, "target_region": region}


def _patch_activity(frame: np.ndarray, grid_size: int = 8) -> np.ndarray:
    gray = np.asarray(frame, dtype=np.float32).mean(axis=-1)
    values = []
    for region_id in range(grid_size * grid_size):
        ys, xs = region_id_to_slice(region_id, gray.shape[0], grid_size)
        values.append(float(gray[ys, xs].mean()))
    return np.asarray(values, dtype=np.float32)


def _render_patch_rollout(template: np.ndarray, activity: np.ndarray, transition: np.ndarray, horizon: int, grid_size: int) -> np.ndarray:
    frames = []
    current = np.asarray(activity, dtype=np.float32)
    base_activity = np.maximum(current, 1e-4)
    for _ in range(horizon):
        current = np.clip(current @ transition, 0.0, 1.0)
        scale = current / base_activity
        frame = template.copy()
        for region_id, value in enumerate(scale):
            ys, xs = region_id_to_slice(region_id, frame.shape[0], grid_size)
            frame[ys, xs] = np.clip(frame[ys, xs] * float(value), 0.0, 1.0)
        frames.append(frame)
    return np.asarray(frames, dtype=np.float32)


def _patch_structure(activity: np.ndarray, accel: np.ndarray, transition: np.ndarray, frame_size: int, grid_size: int) -> dict[str, Any]:
    influence = np.abs(transition) * np.abs(activity[:, None])
    patch_value = np.abs(accel) + 0.5 * influence.mean(axis=0)
    patch_value = patch_value.astype(np.float32)
    event_patch = patch_value > max(0.01, float(patch_value.mean() + patch_value.std()))
    value_map = _patches_to_map(patch_value, frame_size, grid_size)
    event_map = _patches_to_map(event_patch.astype(np.float32), frame_size, grid_size)
    return {
        "applicable": True,
        "patch_activity": activity.astype(np.float32),
        "patch_influence_graph": influence.astype(np.float32),
        "patch_value": patch_value,
        "event_boundary_map": event_map.astype(np.float32),
        "relation_locality_map": value_map.astype(np.float32),
        "inspection_value_field": value_map.astype(np.float32),
        "critical_region_logits": patch_value.copy(),
        "intervention_family": "patch_graph",
    }


def _patches_to_map(values: np.ndarray, frame_size: int, grid_size: int) -> np.ndarray:
    output = np.zeros((frame_size, frame_size), dtype=np.float32)
    for region_id, value in enumerate(values):
        ys, xs = region_id_to_slice(region_id, frame_size, grid_size)
        output[ys, xs] = float(value)
    return output
