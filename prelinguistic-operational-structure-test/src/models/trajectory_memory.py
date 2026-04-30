from __future__ import annotations

from typing import Any

import numpy as np

from ..features import simple_trajectory_extraction
from .base import BasePLOSModel, blank_structure, repeat_last_frame


class TrajectoryMemory(BasePLOSModel):
    name = "trajectory_memory"
    structural_family = "memory"

    def __init__(self) -> None:
        self.fragments: list[np.ndarray] = []

    def fit(self, dataset: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
        self.fragments = [simple_trajectory_extraction(item["past_frames"])["last_centers"] for item in dataset[:128]]

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "future_frames": repeat_last_frame(batch),
            "identity_logits": None,
            "event_logits": None,
            "inspection_logits": None,
            "structure": {**blank_structure(), "memory_size": len(self.fragments)},
        }
