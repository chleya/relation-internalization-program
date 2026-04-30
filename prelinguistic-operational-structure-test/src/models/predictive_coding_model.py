from __future__ import annotations

from typing import Any

import numpy as np

from ..features import patch_mask, region_logits_from_map
from ..model_io import assert_clean_model_batch
from .base import BasePLOSModel
from .field_model import _smooth


class PredictiveCodingModel(BasePLOSModel):
    name = "predictive_coding_model"
    structural_family = "prediction_error"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert_clean_model_batch(batch)
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        future = _pixel_constant_velocity_rollout(past, int(batch["future_horizon"]))
        structure = predictive_coding_structure(past)
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
        if kind not in {"event_latent_perturbation", "inspection_map_shuffle", "inspection_topk_zero", "field_patch_mask", "critical_field_zero"}:
            return {"applicable": False}
        future = output["future_frames"].copy()
        region = int(np.argmax(output["inspection_logits"]))
        mask = patch_mask(future.shape[1], region)[None, :, :, None]
        if kind == "event_latent_perturbation":
            future = np.clip(future - 0.35 * output["structure"]["surprise_field"][None, :, :, None], 0.0, 1.0)
            target = "event_latent_perturbation"
        elif kind == "inspection_map_shuffle":
            future = future * 0.94
            target = "inspection"
        else:
            future = future * (1.0 - 0.40 * mask)
            target = "field"
        return {"applicable": True, "future_frames": future, "base_future_frames": output["future_frames"], "target": target, "target_region": region}


def predictive_coding_structure(past: np.ndarray) -> dict[str, Any]:
    gray = past.mean(axis=-1)
    last = gray[-1]
    prev = gray[-2] if len(gray) > 1 else last
    prev2 = gray[-3] if len(gray) > 2 else prev
    velocity_now = last - prev
    velocity_prev = prev - prev2
    surprise = np.abs(velocity_now - velocity_prev)
    motion_energy = np.abs(velocity_now)
    prediction_error = surprise + 0.5 * motion_energy
    event_threshold = max(0.02, float(prediction_error.mean() + prediction_error.std()))
    event_map = (prediction_error > event_threshold).astype(np.float32)
    locality = _smooth(prediction_error)
    return {
        "applicable": True,
        "prediction_error_field": prediction_error.astype(np.float32),
        "surprise_field": surprise.astype(np.float32),
        "motion_energy_field": motion_energy.astype(np.float32),
        "event_boundary_map": event_map,
        "relation_locality_map": locality.astype(np.float32),
        "inspection_value_field": locality.astype(np.float32),
        "critical_region_logits": region_logits_from_map(locality),
        "intervention_family": "prediction_error",
    }


def _pixel_constant_velocity_rollout(past: np.ndarray, horizon: int) -> np.ndarray:
    last = past[-1]
    prev = past[-2] if len(past) > 1 else last
    velocity = last - prev
    frames = []
    current = last.copy()
    for _ in range(horizon):
        current = np.clip(current + velocity, 0.0, 1.0)
        frames.append(current.copy())
        velocity *= 0.92
    return np.asarray(frames, dtype=np.float32)
