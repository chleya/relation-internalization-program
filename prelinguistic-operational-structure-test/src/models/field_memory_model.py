from __future__ import annotations

from typing import Any

from .base import BasePLOSModel
from ..b23_private_selectors import private_trace_forward, private_trace_intervention, select_region_with_private_selector
from ..b23_private_scorers import field_private_trace_score


class FieldMemoryModel(BasePLOSModel):
    name = "field_memory_model"
    structural_family = "field_memory"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return private_trace_forward(batch, "field_memory")

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        return private_trace_intervention(
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

    def private_trace_scores(self, batch: dict[str, Any]) -> dict[int, float]:
        return field_private_trace_score(self, batch)

    def select_private_trace_region(self, batch: dict[str, Any]) -> dict[str, Any]:
        return select_region_with_private_selector(self, batch)

    def ablate_private_trace(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.intervene_structure(batch, {"type": "force_trace_mask"})

    def disable_shared_selector(self) -> None:
        self.shared_selector_disabled = True
