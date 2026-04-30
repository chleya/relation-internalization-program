from __future__ import annotations

from typing import Any

import numpy as np

from ..features import patch_mask, point_to_region_id, region_logits_from_map
from ..model_io import assert_clean_model_batch
from .base import BasePLOSModel, repeat_last_frame


class FieldModel(BasePLOSModel):
    name = "field_model"
    structural_family = "field"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert_clean_model_batch(batch)
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        structure = field_structure_from_frames(past)
        future = repeat_last_frame(batch)
        return {
            "future_frames": future,
            "identity_logits": None,
            "event_logits": structure["event_boundary_map"],
            "inspection_logits": region_logits_from_map(structure["inspection_value_field"]),
            "structure": structure,
        }

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        output = self.forward(batch)
        if intervention.get("type") not in {"field_patch_mask", "field_patch_swap", "critical_field_zero", "inspection_map_shuffle", "inspection_topk_zero"}:
            return {"applicable": False}
        future = output["future_frames"].copy()
        frame_size = future.shape[1]
        logits = output["inspection_logits"]
        region = int(np.argmax(logits))
        mask = patch_mask(frame_size, region)[None, :, :, None]
        if intervention.get("type") in {"field_patch_mask", "critical_field_zero", "inspection_topk_zero"}:
            future = future * (1.0 - 0.45 * mask)
        elif intervention.get("type") == "field_patch_swap":
            future = np.roll(future, shift=8, axis=2)
        else:
            future = future * 0.92
        return {"applicable": True, "future_frames": future, "base_future_frames": output["future_frames"], "target_region": region}


def field_structure_from_frames(past: np.ndarray) -> dict[str, Any]:
    frame_size = past.shape[1]
    last = past[-1]
    prev = past[-2] if len(past) > 1 else last
    diff = np.abs(last - prev).mean(axis=-1)
    latent = last.mean(axis=-1)
    grad_y, grad_x = np.gradient(latent)
    velocity_field = np.stack([grad_x, grad_y], axis=-1).astype(np.float32)
    accel = np.abs(np.diff(past.mean(axis=-1), axis=0))
    force_response = accel[-3:].mean(axis=0) if len(accel) >= 3 else diff
    uncertainty = (latent < 0.12).astype(np.float32) * 0.15 + diff
    inspection_value = force_response + 0.5 * uncertainty
    event_boundary = (diff > max(0.05, float(diff.mean() + diff.std()))).astype(np.float32)
    relation_locality = _smooth(inspection_value)
    return {
        "applicable": True,
        "latent_field": latent.astype(np.float32),
        "velocity_field": velocity_field,
        "force_response_field": force_response.astype(np.float32),
        "uncertainty_field": uncertainty.astype(np.float32),
        "inspection_value_field": inspection_value.astype(np.float32),
        "event_boundary_map": event_boundary.astype(np.float32),
        "relation_locality_map": relation_locality.astype(np.float32),
        "critical_region_logits": region_logits_from_map(inspection_value),
        "intervention_family": "field",
        "frame_size": frame_size,
    }


def _smooth(value: np.ndarray) -> np.ndarray:
    padded = np.pad(value, 1, mode="edge")
    out = np.zeros_like(value, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += padded[dy : dy + value.shape[0], dx : dx + value.shape[1]]
    return out / 9.0
