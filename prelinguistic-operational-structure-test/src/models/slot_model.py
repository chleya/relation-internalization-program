from __future__ import annotations

from typing import Any

import numpy as np

from ..features import draw_disk, extract_blob_centers, point_to_region_id
from ..model_io import assert_clean_model_batch
from .base import BasePLOSModel, logits_from_region


class SlotModel(BasePLOSModel):
    name = "slot_model"
    structural_family = "slot"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert_clean_model_batch(batch)
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        future_horizon = int(batch["future_horizon"])
        centers = extract_blob_centers(past[-1])
        prev = extract_blob_centers(past[-2]) if len(past) > 1 else centers
        velocity = _match_velocity(centers, prev)
        future = _render_rollout(centers, velocity, future_horizon, past.shape[1], past[-1])
        region = int(point_to_region_id(centers.mean(axis=0), past.shape[1])) if len(centers) else 0
        structure = {
            "applicable": True,
            "slots": centers,
            "slot_attention": np.ones(len(centers), dtype=np.float32),
            "interaction_edges": _pair_edges(centers),
            "critical_region_logits": logits_from_region(region),
        }
        return {
            "future_frames": future,
            "identity_logits": np.ones((2, 2), dtype=np.float32),
            "event_logits": _event_logits_from_centers(centers, past.shape[1]),
            "inspection_logits": structure["critical_region_logits"],
            "structure": structure,
        }

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        output = self.forward(batch)
        if intervention.get("type") not in {"slot_removal", "slot_swap", "slot_noise"}:
            return {"applicable": False}
        changed = output["future_frames"].copy()
        if intervention.get("type") == "slot_removal":
            changed[:, :, :, :] *= 0.55
        elif intervention.get("type") == "slot_swap":
            changed = changed[:, :, ::-1, :]
        else:
            changed = np.clip(changed + 0.08, 0.0, 1.0)
        return {"applicable": True, "future_frames": changed, "base_future_frames": output["future_frames"], "target": "slot"}


def _match_velocity(centers: np.ndarray, prev: np.ndarray) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(prev) != len(centers):
        return np.zeros_like(centers)
    return centers - prev


def _render_rollout(centers: np.ndarray, velocity: np.ndarray, horizon: int, frame_size: int, template: np.ndarray) -> np.ndarray:
    future = []
    colors = [np.asarray([0.95, 0.25, 0.25], dtype=np.float32), np.asarray([0.25, 0.65, 1.0], dtype=np.float32)]
    pos = centers.copy()
    vel = velocity.copy()
    for _ in range(horizon):
        frame = np.zeros_like(template)
        for idx, center in enumerate(pos[:2]):
            draw_disk(frame, center, 4.0, colors[idx % len(colors)])
        future.append(frame)
        pos = pos + vel
        for axis in (0, 1):
            hit = (pos[:, axis] < 4.0) | (pos[:, axis] > frame_size - 5.0)
            vel[hit, axis] *= -1
            pos[:, axis] = np.clip(pos[:, axis], 4.0, frame_size - 5.0)
    return np.asarray(future, dtype=np.float32)


def _pair_edges(centers: np.ndarray) -> np.ndarray:
    if len(centers) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray([[0, 1], [1, 0]], dtype=np.int64)


def _event_logits_from_centers(centers: np.ndarray, frame_size: int) -> np.ndarray:
    logits = np.zeros((frame_size, frame_size), dtype=np.float32)
    if len(centers) >= 2:
        point = centers.mean(axis=0).astype(int)
        x, y = np.clip(point, 0, frame_size - 1)
        logits[max(0, y - 3) : min(frame_size, y + 4), max(0, x - 3) : min(frame_size, x + 4)] = 1.0
    return logits
