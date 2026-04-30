from __future__ import annotations

from typing import Any

from .base import BasePLOSModel, blank_structure, repeat_last_frame


class PixelPredictor(BasePLOSModel):
    name = "pixel_predictor"
    structural_family = "pixel"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "future_frames": repeat_last_frame(batch),
            "identity_logits": None,
            "event_logits": None,
            "inspection_logits": None,
            "structure": blank_structure(),
        }
