from __future__ import annotations

from typing import Any

from .base import BasePLOSModel
from .delayed_common import delayed_trace_forward, delayed_trace_intervention


class FieldMemoryModel(BasePLOSModel):
    name = "field_memory_model"
    structural_family = "field_memory"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return delayed_trace_forward(batch, "field_memory")

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        return delayed_trace_intervention(
            batch,
            intervention,
            family="field_memory",
            causal_types={
                "force_trace_mask",
                "uncertainty_trace_zero",
                "delayed_influence_patch_swap",
                "field_memory_shuffle",
            },
        )
