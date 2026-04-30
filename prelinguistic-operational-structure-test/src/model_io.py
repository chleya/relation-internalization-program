from __future__ import annotations

from typing import Any

import numpy as np


FORBIDDEN_MODEL_INPUT_KEYS = {
    "future_frames",
    "ground_truth",
    "true_positions",
    "true_velocities",
    "true_object_ids",
    "object_ids",
    "event_labels",
    "relation_labels",
    "language",
    "rules",
    "relation_table",
    "episode_type",
}


def make_model_batch(episode: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    past = np.asarray(episode["past_frames"], dtype=np.float32)
    env = (config or {}).get("env", {})
    batch = {
        "past_frames": past.copy(),
        "future_horizon": int(np.asarray(episode["future_frames"]).shape[0]),
        "frame_size": int(env.get("frame_size", past.shape[1])),
        "grid_size": int(env.get("grid_size", 8)),
    }
    assert_clean_model_batch(batch)
    return batch


def assert_clean_model_batch(batch: dict[str, Any]) -> None:
    forbidden = sorted(FORBIDDEN_MODEL_INPUT_KEYS.intersection(batch))
    if forbidden:
        raise ValueError(f"model input contains forbidden keys: {', '.join(forbidden)}")
    if "past_frames" not in batch:
        raise ValueError("model input must contain past_frames")
    if "future_horizon" not in batch:
        raise ValueError("model input must contain future_horizon")
