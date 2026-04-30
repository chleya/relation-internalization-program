from __future__ import annotations

from typing import Any

import numpy as np

from ..features import extract_blob_centers
from ..model_io import assert_clean_model_batch
from .base import BasePLOSModel, repeat_last_frame


class WorldModel(BasePLOSModel):
    name = "world_model"
    structural_family = "latent"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert_clean_model_batch(batch)
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        centers = extract_blob_centers(past[-1])
        latent = centers.flatten() if len(centers) else np.zeros(4, dtype=np.float32)
        return {
            "future_frames": repeat_last_frame(batch),
            "identity_logits": None,
            "event_logits": None,
            "inspection_logits": None,
            "structure": {"applicable": True, "latent_state": latent, "intervention_family": "global_latent"},
        }

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        output = self.forward(batch)
        if intervention.get("type") != "event_latent_perturbation":
            return {"applicable": False}
        changed = output["future_frames"].copy()
        changed *= 0.90
        return {"applicable": True, "future_frames": changed, "base_future_frames": output["future_frames"]}
