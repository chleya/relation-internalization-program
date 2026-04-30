from __future__ import annotations

from typing import Any

import numpy as np

from ..model_io import assert_clean_model_batch


class BasePLOSModel:
    name = "base"
    structural_family = "none"

    def fit(self, dataset: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
        return None

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def get_structure(self, batch: dict[str, Any]) -> dict[str, Any]:
        output = self.forward(batch)
        return output.get("structure", {"applicable": False})

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        return {"applicable": False, "reason": "no_intervenable_structure"}


def repeat_last_frame(batch: dict[str, Any]) -> np.ndarray:
    assert_clean_model_batch(batch)
    past = np.asarray(batch["past_frames"], dtype=np.float32)
    horizon = int(batch["future_horizon"])
    return np.repeat(past[-1][None, ...], horizon, axis=0)


def blank_structure() -> dict[str, Any]:
    return {"applicable": False}


def logits_from_region(region_id: int, n_regions: int = 64, strength: float = 4.0) -> np.ndarray:
    logits = np.zeros(n_regions, dtype=np.float32)
    logits[int(region_id)] = float(strength)
    return logits
