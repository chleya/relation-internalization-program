from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .b2_delayed_env import choose_far_region, draw_trace
from .features import region_id_to_slice
from .model_io import make_model_batch


TRACE_INTERVENTIONS = {
    "recurrent_flow_checkpoint": "memory_trace_zero",
    "field_memory": "force_trace_mask",
    "schema_memory": "delayed_candidate_zero",
}


def trace_family(model: Any, batch: dict[str, Any] | None = None) -> str:
    family = str(getattr(model, "structural_family", ""))
    if family in TRACE_INTERVENTIONS:
        return family
    if batch is not None:
        structure = model.get_structure(batch)
        return str(structure.get("trace_family", structure.get("intervention_family", "")))
    return family


def apply_true_trace_deletion(model: Any, batch: dict[str, Any], trace_family_name: str | None = None) -> dict[str, Any]:
    family = trace_family_name or trace_family(model, batch)
    intervention = TRACE_INTERVENTIONS.get(family)
    if intervention is None:
        return {"applicable": False, "reason": "unsupported_trace_family"}
    return model.intervene_structure(batch, {"type": intervention})


def apply_matched_non_trace_deletion(model: Any, batch: dict[str, Any], trace_family_name: str | None = None) -> dict[str, Any]:
    family = trace_family_name or trace_family(model, batch)
    if family not in TRACE_INTERVENTIONS:
        return {"applicable": False, "reason": "unsupported_trace_family"}
    return model.intervene_structure(batch, {"type": "non_trace_control"})


def apply_trace_swap(model: Any, batch_a: dict[str, Any], batch_b: dict[str, Any], trace_family_name: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    family = trace_family_name or trace_family(model, batch_a)
    if family not in TRACE_INTERVENTIONS:
        return {"applicable": False, "reason": "unsupported_trace_family"}, {"applicable": False, "reason": "unsupported_trace_family"}
    output_a = model.forward(batch_a)
    output_b = model.forward(batch_b)
    return {
        "applicable": True,
        "base_output": output_a,
        "swapped_output": model.forward(batch_b),
    }, {
        "applicable": True,
        "base_output": output_b,
        "swapped_output": model.forward(batch_a),
    }


def apply_trace_compression(
    model: Any,
    batch: dict[str, Any],
    compression_level: float,
    trace_family_name: str | None = None,
) -> dict[str, Any]:
    family = trace_family_name or trace_family(model, batch)
    if family not in TRACE_INTERVENTIONS:
        return {"applicable": False, "reason": "unsupported_trace_family"}
    compressed_batch = dict(batch)
    past = np.asarray(batch["past_frames"], dtype=np.float32).copy()
    level = float(np.clip(compression_level, 0.0, 1.0))
    trace_mask = (past.mean(axis=-1) > 0.035) & (past.max(axis=-1) < 0.19)
    past[trace_mask] *= level
    compressed_batch["past_frames"] = past
    return {
        "applicable": True,
        "compression_level": level,
        "base_output": model.forward(batch),
        "compressed_output": model.forward(compressed_batch),
    }


def mask_region_in_episode(episode: dict[str, Any], region: int, config: dict[str, Any], fill: float = 0.0) -> dict[str, Any]:
    altered = copy.deepcopy(episode)
    env = config.get("env", {})
    frame_size = int(env.get("frame_size", altered["past_frames"].shape[1]))
    grid_size = int(env.get("grid_size", 8))
    ys, xs = region_id_to_slice(region, frame_size, grid_size)
    past = np.asarray(altered["past_frames"], dtype=np.float32).copy()
    past[:, ys, xs, :] = fill
    altered["past_frames"] = past
    frames = np.asarray(altered["frames"], dtype=np.float32).copy()
    frames[: len(past)] = past
    altered["frames"] = frames
    return altered


def swap_trace_regions_between_episodes(
    episode_a: dict[str, Any],
    episode_b: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    env = config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    a_true = int(episode_a["ground_truth"].get("true_trace_region", episode_a["ground_truth"].get("true_delayed_checkpoint_region")))
    b_true = int(episode_b["ground_truth"].get("true_trace_region", episode_b["ground_truth"].get("true_delayed_checkpoint_region")))
    swapped_a = mask_region_in_episode(episode_a, a_true, config, fill=0.0)
    swapped_b = mask_region_in_episode(episode_b, b_true, config, fill=0.0)
    for frame in swapped_a["past_frames"]:
        draw_trace(frame, b_true, frame_size, grid_size, 1.0)
    for frame in swapped_b["past_frames"]:
        draw_trace(frame, a_true, frame_size, grid_size, 1.0)
    swapped_a["frames"][: len(swapped_a["past_frames"])] = swapped_a["past_frames"]
    swapped_b["frames"][: len(swapped_b["past_frames"])] = swapped_b["past_frames"]
    swapped_a["ground_truth"]["swapped_trace_region"] = b_true
    swapped_b["ground_truth"]["swapped_trace_region"] = a_true
    return swapped_a, swapped_b


def add_false_trace_to_episode(episode: dict[str, Any], false_region: int, config: dict[str, Any], strength: float = 0.72) -> dict[str, Any]:
    altered = copy.deepcopy(episode)
    env = config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    for frame in altered["past_frames"]:
        draw_trace(frame, false_region, frame_size, grid_size, strength)
    altered["frames"][: len(altered["past_frames"])] = altered["past_frames"]
    altered["ground_truth"]["false_trace_region"] = int(false_region)
    return altered


def choose_matched_non_trace_region(true_region: int, grid_size: int, seed: int) -> int:
    return choose_far_region(true_region, grid_size, seed)


def make_model_batch_for_episode(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return make_model_batch(episode, config)
