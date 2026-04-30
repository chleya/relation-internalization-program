from __future__ import annotations

from typing import Any

from .base import BasePLOSModel
from .delayed_common import delayed_trace_forward, delayed_trace_intervention


class RecurrentFlowCheckpointModel(BasePLOSModel):
    name = "recurrent_flow_checkpoint_model"
    structural_family = "recurrent_flow_checkpoint"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return delayed_trace_forward(batch, "recurrent_flow_checkpoint")

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        return delayed_trace_intervention(
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
