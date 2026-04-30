from __future__ import annotations

from typing import Any

import numpy as np

from .b23_private_scorers import private_trace_scores, selected_region
from .model_io import assert_clean_model_batch


SOURCE_FAMILY = {
    "recurrent_flow_checkpoint": "recurrent_memory",
    "field_memory": "field_trace",
    "schema_memory": "schema_memory",
}

SOURCE_MODULE = {
    "recurrent_flow_checkpoint": "b23_private_scorers.recurrent_private_trace_score",
    "field_memory": "b23_private_scorers.field_private_trace_score",
    "schema_memory": "b23_private_scorers.schema_private_trace_score",
}


def private_trace_forward(batch: dict[str, Any], family: str) -> dict[str, Any]:
    from .models.delayed_common import build_delayed_structure, estimate_delayed_force, extract_trace_field, trace_candidates
    from .models.flow_checkpoint_model import _rollout_tracks, _track_components, _track_motion_energy

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
            "structure": {
                "applicable": False,
                "shared_selector_used": False,
                "model_private_score_used": False,
                "fallback_used": True,
                "fallback_reason": "insufficient_motion",
            },
        }

    trace_field = extract_trace_field(past)
    candidates = trace_candidates(trace_field, frame_size, grid_size)
    scores = private_trace_scores(batch, family)
    if scores:
        region = selected_region(scores)
        confidence = float(max(scores.values()))
        fallback_used = False
        fallback_reason = ""
    else:
        region = int(point_region(tracks["positions"][-1].mean(axis=0), frame_size, grid_size))
        confidence = 0.0
        fallback_used = True
        fallback_reason = "no_private_trace_candidate"

    checkpoint = {
        "region": int(region),
        "kind": "private_delayed_trace",
        "force_region": int(region),
        "force_vector": estimate_delayed_force(tracks),
        "collision_region": None,
        "occluder_region": None,
    }
    future = _rollout_tracks(tracks, checkpoint, horizon, frame_size, past[-1])
    structure = build_delayed_structure(
        family=family,
        trace_field=trace_field,
        candidates=candidates,
        selected_region=int(region),
        confidence=confidence,
        frame_size=frame_size,
        grid_size=grid_size,
        tracks=tracks,
    )
    structure.update(
        {
            "private_selector_stage": "B2.3",
            "private_trace_scores": {int(key): float(value) for key, value in scores.items()},
            "candidate_scores": {int(key): float(value) for key, value in scores.items()},
            "selected_region": int(region),
            "source_trace_family": SOURCE_FAMILY[family],
            "source_module": SOURCE_MODULE[family],
            "shared_selector_used": False,
            "model_private_score_used": not fallback_used,
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
        }
    )
    return {
        "future_frames": future,
        "identity_logits": np.eye(2, dtype=np.float32),
        "event_logits": structure.get("event_boundary_map", structure.get("inspection_value_field")),
        "inspection_logits": structure["inspection_logits"],
        "structure": structure,
    }


def private_trace_intervention(batch: dict[str, Any], intervention: dict[str, Any], family: str, causal_types: set[str]) -> dict[str, Any]:
    output = private_trace_forward(batch, family)
    if not output.get("structure", {}).get("applicable", False):
        return {"applicable": False, "reason": "no_private_trace_structure"}
    kind = str(intervention.get("type", ""))
    if kind == "non_trace_control":
        future = output["future_frames"].copy()
        return {
            "applicable": True,
            "future_frames": future,
            "base_future_frames": output["future_frames"],
            "target": "non_trace_control",
            "target_region": int(np.argmax(output["inspection_logits"])),
        }
    if kind not in causal_types and kind not in {"private_trace_zero", "selector_free_trace_zero"}:
        return {"applicable": False, "reason": "unsupported_private_trace_intervention"}
    altered = dict(batch)
    past = np.asarray(batch["past_frames"], dtype=np.float32).copy()
    trace_mask = (past.mean(axis=-1) > 0.035) & (past.max(axis=-1) < 0.19)
    past[trace_mask] = 0.0
    altered["past_frames"] = past
    future = private_trace_forward(altered, family)["future_frames"]
    return {
        "applicable": True,
        "future_frames": future,
        "base_future_frames": output["future_frames"],
        "target": SOURCE_FAMILY[family],
        "target_region": int(np.argmax(output["inspection_logits"])),
        "intervention_type": kind,
    }


def enforce_private_selector(model: Any, model_name: str) -> None:
    if hasattr(model, "disable_shared_selector"):
        model.disable_shared_selector()
    setattr(model, "private_selector_required", True)
    setattr(model, "shared_selector_disabled", True)


def get_private_trace_scores(model: Any, batch: dict[str, Any]) -> dict[int, float]:
    if hasattr(model, "private_trace_scores"):
        return dict(model.private_trace_scores(batch))
    return private_trace_scores(batch, str(getattr(model, "structural_family", "")))


def select_region_with_private_selector(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    scores = get_private_trace_scores(model, batch)
    region = selected_region(scores)
    family = str(getattr(model, "structural_family", ""))
    return {
        "selected_region": int(region),
        "score": float(scores.get(region, 0.0)),
        "source_trace_family": SOURCE_FAMILY.get(family, "unknown"),
        "source_module": SOURCE_MODULE.get(family, "unknown"),
        "shared_selector_used": False,
        "fallback_used": not bool(scores),
        "model_private_score_used": bool(scores),
        "candidate_scores": scores,
    }


def validate_no_shared_selector_call(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    output = model.forward(batch)
    structure = output.get("structure", {})
    shared = bool(structure.get("shared_selector_used", False))
    return {
        "shared_selector_used": shared,
        "pass": int(not shared),
        "selected_region": int(structure.get("selected_region", np.argmax(output.get("inspection_logits", [0])))),
    }


def point_region(point: np.ndarray, frame_size: int, grid_size: int) -> int:
    cell = frame_size / grid_size
    gx = min(grid_size - 1, max(0, int(float(point[0]) // cell)))
    gy = min(grid_size - 1, max(0, int(float(point[1]) // cell)))
    return int(gy * grid_size + gx)
