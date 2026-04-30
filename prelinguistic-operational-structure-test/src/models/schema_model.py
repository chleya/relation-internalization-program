from __future__ import annotations

from typing import Any

import numpy as np

from ..features import extract_blob_centers, point_to_region_id, region_logits_from_map
from ..model_io import assert_clean_model_batch
from .field_model import FieldModel, field_structure_from_frames
from .slot_model import _pair_edges, _render_rollout


class SchemaModel(FieldModel):
    name = "schema_model"
    structural_family = "field_first_schema"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert_clean_model_batch(batch)
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        future_horizon = int(batch["future_horizon"])
        field_structure = field_structure_from_frames(past)
        centers = extract_blob_centers(past[-1])
        prev = extract_blob_centers(past[-2]) if len(past) > 1 else centers
        velocity = centers - prev if len(centers) and len(prev) == len(centers) else np.zeros_like(centers)
        if len(centers):
            future = _render_rollout(centers, velocity, future_horizon, past.shape[1], past[-1])
        else:
            future = np.repeat(past[-1][None, ...], future_horizon, axis=0)
        critical_logits = field_structure["critical_region_logits"].copy()
        if len(centers) >= 2:
            contact_region = point_to_region_id(centers.mean(axis=0), past.shape[1])
            critical_logits[int(contact_region)] += 1.0
        structure = {
            **field_structure,
            "event_boundary_map": np.maximum(field_structure["event_boundary_map"], field_structure["relation_locality_map"] > 0.08).astype(np.float32),
            "relation_locality_map": field_structure["relation_locality_map"],
            "critical_region_logits": critical_logits,
            "object_candidates": centers,
            "interaction_edges": _pair_edges(centers),
            "intervention_family": "field_first_schema",
        }
        return {
            "future_frames": future,
            "identity_logits": np.ones((2, 2), dtype=np.float32) if len(centers) >= 2 else None,
            "event_logits": structure["event_boundary_map"],
            "inspection_logits": critical_logits,
            "structure": structure,
        }

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        output = self.forward(batch)
        kind = intervention.get("type")
        if kind in {"field_patch_mask", "field_patch_swap", "critical_field_zero", "inspection_map_shuffle", "inspection_topk_zero"}:
            return super().intervene_structure(batch, intervention)
        if kind in {"event_latent_perturbation", "relation_edge_ablation"}:
            changed = output["future_frames"].copy() * 0.75
            return {"applicable": True, "future_frames": changed, "base_future_frames": output["future_frames"], "target": kind}
        if kind in {"slot_removal", "slot_swap", "slot_noise"} and len(output["structure"].get("object_candidates", [])):
            changed = output["future_frames"].copy() * 0.70
            return {"applicable": True, "future_frames": changed, "base_future_frames": output["future_frames"], "target": kind}
        return {"applicable": False}
