from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np

from .model_io import assert_clean_model_batch


NULL_CONTROL_COLS = [
    "null_event_activation",
    "null_inspection_concentration",
    "null_structure_applicable",
    "null_prior_alarm",
]


def evaluate_null_controls(model: Any, config: dict[str, Any], seed: int = 0) -> dict[str, float]:
    rows = []
    for mode in ("blank", "static_low_noise"):
        batch = make_null_batch(config, seed, mode)
        output = model.forward(batch)
        rows.append(null_control_metrics(output))
    return {key: mean(float(row[key]) for row in rows) for key in NULL_CONTROL_COLS}


def make_null_batch(config: dict[str, Any], seed: int, mode: str) -> dict[str, Any]:
    env = config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    past_horizon = int(env.get("past_frames", 8))
    future_horizon = int(env.get("future_frames", 12))
    grid_size = int(env.get("grid_size", 8))
    rng = np.random.default_rng(seed)
    if mode == "blank":
        frame = np.zeros((frame_size, frame_size, 3), dtype=np.float32)
    elif mode == "static_low_noise":
        frame = rng.random((frame_size, frame_size, 3), dtype=np.float32) * 0.05
    else:
        raise ValueError(f"unknown null control mode: {mode}")
    batch = {
        "past_frames": np.repeat(frame[None, ...], past_horizon, axis=0),
        "future_horizon": future_horizon,
        "frame_size": frame_size,
        "grid_size": grid_size,
    }
    assert_clean_model_batch(batch)
    return batch


def null_control_metrics(output: dict[str, Any]) -> dict[str, float]:
    event = output.get("event_logits")
    event_activation = float(np.asarray(event, dtype=np.float32).mean()) if event is not None else 0.0
    inspection = output.get("inspection_logits")
    concentration = _inspection_concentration(inspection)
    applicable = 1.0 if output.get("structure", {}).get("applicable", False) else 0.0
    alarm = 1.0 if event_activation > 0.05 or concentration > 0.25 else 0.0
    return {
        "null_event_activation": event_activation,
        "null_inspection_concentration": concentration,
        "null_structure_applicable": applicable,
        "null_prior_alarm": alarm,
    }


def _inspection_concentration(logits: Any) -> float:
    if logits is None:
        return 0.0
    values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    shifted = values - float(values.max())
    exp = np.exp(shifted)
    probs = exp / max(float(exp.sum()), 1e-8)
    return float(probs.max() - (1.0 / values.size))
