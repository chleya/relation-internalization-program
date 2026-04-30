from __future__ import annotations

from typing import Any

from .base import BasePLOSModel
from ..b23_private_selectors import private_trace_forward, private_trace_intervention, select_region_with_private_selector
from ..b23_private_scorers import recurrent_private_trace_score


class RecurrentFlowCheckpointModel(BasePLOSModel):
    name = "recurrent_flow_checkpoint_model"
    structural_family = "recurrent_flow_checkpoint"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return private_trace_forward(batch, "recurrent_flow_checkpoint")

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        return private_trace_intervention(
            batch,
            intervention,
            family="recurrent_flow_checkpoint",
            causal_types={
                "memory_trace_zero",
                "memory_trace_shuffle",
                "delayed_checkpoint_map_zero",
                "checkpoint_logits_swap",
            },
        )

    def private_trace_scores(self, batch: dict[str, Any]) -> dict[int, float]:
        return recurrent_private_trace_score(self, batch)

    def select_private_trace_region(self, batch: dict[str, Any]) -> dict[str, Any]:
        return select_region_with_private_selector(self, batch)

    def ablate_private_trace(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.intervene_structure(batch, {"type": "memory_trace_zero"})

    def disable_shared_selector(self) -> None:
        self.shared_selector_disabled = True
