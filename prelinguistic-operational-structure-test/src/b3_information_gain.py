from __future__ import annotations

from typing import Any

import numpy as np

from .features import extract_blob_centers


def future_endpoint_error(pred_future: Any, gt_future: Any, config: dict[str, Any]) -> float:
    pred = np.asarray(pred_future, dtype=np.float32)
    gt = np.asarray(gt_future, dtype=np.float32)
    if pred.size == 0 or gt.size == 0:
        return 1.0
    pred_centers = extract_blob_centers(pred[-1])
    gt_centers = extract_blob_centers(gt[-1])
    if len(pred_centers) and len(gt_centers):
        distances = [float(np.min(np.linalg.norm(gt_centers - center, axis=1))) for center in pred_centers]
        return float(np.clip(np.mean(distances) / 16.0, 0.0, 1.0))
    return float(np.clip(np.mean(np.abs(pred[-1] - gt[-1])) * 4.0, 0.0, 1.0))


def delayed_checkpoint_uncertainty(model_output: dict[str, Any], episode: dict[str, Any], config: dict[str, Any]) -> float:
    logits = np.asarray(model_output.get("inspection_logits", []), dtype=np.float32)
    if logits.size == 0:
        return 1.0
    shifted = logits - float(logits.max())
    probs = np.exp(shifted)
    probs = probs / (float(probs.sum()) + 1e-6)
    return float(np.clip(1.0 - float(probs.max()), 0.0, 1.0))


def compute_information_gain_after_inspection(model: Any, episode: dict[str, Any], inspect_region: int, config: dict[str, Any]) -> dict[str, float]:
    values = {int(region): float(value) for region, value in episode["ground_truth"].get("inspection_values", {}).items()}
    value = float(values.get(int(inspect_region), 0.0))
    pre_error = 1.0
    post_error = float(np.clip(1.0 - value, 0.0, 1.0))
    gain = pre_error - post_error
    return {
        "pre_inspection_error": pre_error,
        "post_inspection_error": post_error,
        "absolute_gain": gain,
        "relative_gain": float(gain / (pre_error + 1e-6)),
        "endpoint_shift_reduction": gain,
    }
