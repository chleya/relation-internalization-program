from __future__ import annotations

from typing import Any

from .base import BasePLOSModel
from .delayed_common import delayed_trace_forward, delayed_trace_intervention


class SchemaMemoryModel(BasePLOSModel):
    name = "schema_memory_model"
    structural_family = "schema_memory"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return delayed_trace_forward(batch, "schema_memory")

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        return delayed_trace_intervention(
            batch,
            intervention,
            family="schema_memory",
            causal_types={
                "schema_slot_removal",
                "schema_slot_swap",
                "delayed_candidate_zero",
                "schema_delay_logits_shuffle",
            },
        )
