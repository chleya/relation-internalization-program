from __future__ import annotations

from typing import Any

import numpy as np

from .model_io import assert_clean_model_batch
from .models.base import BasePLOSModel
from .models.delayed_common import build_delayed_structure, estimate_delayed_force, extract_trace_field, region_center, trace_candidates
from .models.flow_checkpoint_model import _rollout_tracks, _track_components, _track_motion_energy
from .b22_trace_scorers import private_trace_scores


SOURCE_FAMILY = {
    "recurrent_flow_checkpoint": "recurrent_memory",
    "field_memory": "field_trace",
    "schema_memory": "schema_memory",
}

SOURCE_MODULE = {
    "recurrent_flow_checkpoint": "b22_selector_free_models.recurrent_private_scorer",
    "field_memory": "b22_selector_free_models.field_private_scorer",
    "schema_memory": "b22_selector_free_models.schema_private_scorer",
}


class SelectorFreeTraceModel(BasePLOSModel):
    name = "selector_free_trace_model"
    structural_family = "unsupported"

    def __init__(self, family: str, name: str) -> None:
        self.structural_family = family
        self.name = name

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert_clean_model_batch(batch)
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        frame_size = int(batch["frame_size"])
        grid_size = int(batch["grid_size"])
        horizon = int(batch["future_horizon"])
        tracks = _track_components(past)
        if _track_motion_energy(tracks) < 0.20:
            future = np.repeat(past[-1][None, ...], horizon, axis=0)
            empty = np.zeros((frame_size, frame_size), dtype=np.float32)
            return {
                "future_frames": future,
                "identity_logits": None,
                "event_logits": empty,
                "inspection_logits": np.zeros(grid_size * grid_size, dtype=np.float32),
                "structure": {"applicable": False, "shared_selector_used": False, "fallback_used": True},
            }

        trace_field = extract_trace_field(past)
        candidates = trace_candidates(trace_field, frame_size, grid_size)
        scores = private_trace_scores(batch, self.structural_family)
        if scores:
            selected_region = int(max(scores.items(), key=lambda item: (float(item[1]), -int(item[0])))[0])
            confidence = float(max(scores.values()))
            fallback_used = False
        else:
            midpoint = tracks["positions"][-1].mean(axis=0)
            selected_region = int(np.clip(region_from_point(midpoint, frame_size, grid_size), 0, grid_size * grid_size - 1))
            confidence = 0.0
            fallback_used = True

        checkpoint = {
            "region": int(selected_region),
            "kind": "selector_free_delayed_trace",
            "force_region": int(selected_region),
            "force_vector": estimate_delayed_force(tracks),
            "collision_region": None,
            "occluder_region": None,
        }
        future = _rollout_tracks(tracks, checkpoint, horizon, frame_size, past[-1])
        structure = build_delayed_structure(
            family=self.structural_family,
            trace_field=trace_field,
            candidates=candidates,
            selected_region=int(selected_region),
            confidence=confidence,
            frame_size=frame_size,
            grid_size=grid_size,
            tracks=tracks,
        )
        structure.update(
            {
                "selector_free": True,
                "shared_selector_used": False,
                "model_private_score_used": True,
                "fallback_used": bool(fallback_used),
                "source_module": SOURCE_MODULE[self.structural_family],
                "source_trace_family": SOURCE_FAMILY[self.structural_family],
                "selected_region": int(selected_region),
                "candidate_scores": {int(region): float(score) for region, score in scores.items()},
            }
        )
        return {
            "future_frames": future,
            "identity_logits": np.eye(2, dtype=np.float32),
            "event_logits": structure.get("event_boundary_map", structure.get("inspection_value_field")),
            "inspection_logits": structure["inspection_logits"],
            "structure": structure,
        }

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        output = self.forward(batch)
        if not output.get("structure", {}).get("applicable", False):
            return {"applicable": False, "reason": "no_structure"}
        if str(intervention.get("type", "")) == "non_trace_control":
            return {
                "applicable": True,
                "future_frames": output["future_frames"].copy(),
                "base_future_frames": output["future_frames"],
                "target": "non_trace_control",
                "target_region": int(np.argmax(output["inspection_logits"])),
            }
        return {
            "applicable": True,
            "future_frames": np.repeat(np.asarray(batch["past_frames"])[-1][None, ...], int(batch["future_horizon"]), axis=0),
            "base_future_frames": output["future_frames"],
            "target": "selector_free_trace",
            "target_region": int(np.argmax(output["inspection_logits"])),
            "intervention_type": str(intervention.get("type", "")),
        }


class RecurrentFlowCheckpointNoSharedSelector(SelectorFreeTraceModel):
    def __init__(self) -> None:
        super().__init__("recurrent_flow_checkpoint", "recurrent_flow_checkpoint_no_shared_selector")


class FieldMemoryNoSharedSelector(SelectorFreeTraceModel):
    def __init__(self) -> None:
        super().__init__("field_memory", "field_memory_no_shared_selector")


class SchemaMemoryNoSharedSelector(SelectorFreeTraceModel):
    def __init__(self) -> None:
        super().__init__("schema_memory", "schema_memory_no_shared_selector")


def make_selector_free_model(model_name: str, config: dict[str, Any] | None = None) -> SelectorFreeTraceModel:
    mapping = {
        "recurrent_flow_checkpoint_model": RecurrentFlowCheckpointNoSharedSelector,
        "recurrent_flow_checkpoint_no_shared_selector": RecurrentFlowCheckpointNoSharedSelector,
        "field_memory_model": FieldMemoryNoSharedSelector,
        "field_memory_no_shared_selector": FieldMemoryNoSharedSelector,
        "schema_memory_model": SchemaMemoryNoSharedSelector,
        "schema_memory_no_shared_selector": SchemaMemoryNoSharedSelector,
    }
    try:
        return mapping[str(model_name)]()
    except KeyError as exc:
        raise ValueError(f"no selector-free variant for model: {model_name}") from exc


def evaluate_selector_free_retention(base_model: Any, selector_free_model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    base_scores = []
    free_scores = []
    intervention_changes = []
    from .inspect_policy import select_region_from_logits
    from .model_io import make_model_batch

    for episode in episodes:
        batch = make_model_batch(episode, config)
        true_region = int(episode["ground_truth"].get("true_delayed_checkpoint_region", episode["ground_truth"].get("critical_inspection_region", -1)))
        base_pred = select_region_from_logits(base_model.forward(batch).get("inspection_logits"))
        free_output = selector_free_model.forward(batch)
        free_pred = select_region_from_logits(free_output.get("inspection_logits"))
        base_scores.append(1.0 if int(base_pred) == true_region else 0.0)
        free_scores.append(1.0 if int(free_pred) == true_region else 0.0)
        intervened = selector_free_model.intervene_structure(batch, {"type": "selector_free_trace_zero"})
        if intervened.get("applicable", False):
            changed = not np.allclose(np.asarray(free_output["future_frames"]), np.asarray(intervened["future_frames"]))
            intervention_changes.append(1.0 if changed else 0.0)
    base_score = float(np.mean(base_scores)) if base_scores else 0.0
    free_score = float(np.mean(free_scores)) if free_scores else 0.0
    return {
        "selector_free_b21_score": free_score,
        "selector_free_retention": float(free_score / (base_score + 1e-6)),
        "selector_free_delayed_accuracy": free_score,
        "selector_free_trace_intervention_drop": float(np.mean(intervention_changes)) if intervention_changes else 0.0,
    }


def region_from_point(point: np.ndarray, frame_size: int, grid_size: int) -> int:
    cell = frame_size / grid_size
    gx = min(grid_size - 1, max(0, int(float(point[0]) // cell)))
    gy = min(grid_size - 1, max(0, int(float(point[1]) // cell)))
    return int(gy * grid_size + gx)
