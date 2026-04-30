from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np

from .b2_delayed_metrics import delayed_endpoint_shift
from .model_io import make_model_batch


TRACE_INTERVENTIONS = {
    "recurrent_flow_checkpoint": "memory_trace_zero",
    "field_memory": "force_trace_mask",
    "schema_memory": "delayed_candidate_zero",
}


def apply_causal_trace_intervention(model: Any, batch: dict[str, Any], intervention_type: str) -> dict[str, Any]:
    """
    Apply model-specific delayed trace intervention.
    """

    return model.intervene_structure(batch, {"type": intervention_type})


def apply_non_trace_control_intervention(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """
    Perturb a non-delayed or noncausal structure of similar size.
    """

    return model.intervene_structure(batch, {"type": "non_trace_control"})


def evaluate_causal_trace_intervention(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    return evaluate_causal_trace_intervention_with_records(model, episodes, config)[0]


def evaluate_causal_trace_intervention_with_records(
    model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    family = getattr(model, "structural_family", "")
    intervention_type = TRACE_INTERVENTIONS.get(str(family), "unsupported_delayed_trace")
    causal_shifts: list[float] = []
    control_shifts: list[float] = []
    records: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        batch = make_model_batch(episode, config)
        causal = apply_causal_trace_intervention(model, batch, intervention_type)
        control = apply_non_trace_control_intervention(model, batch)
        if not causal.get("applicable", False) or not control.get("applicable", False):
            records.append(
                {
                    "episode_id": idx,
                    "episode_type": episode["ground_truth"].get("episode_type", ""),
                    "delay": episode["ground_truth"].get("delay", ""),
                    "true_delayed_checkpoint_region": episode["ground_truth"].get("true_delayed_checkpoint_region", ""),
                    "early_saliency_region": episode["ground_truth"].get("early_saliency_region", ""),
                    "predicted_region": "",
                    "selected_delay": "",
                    "correct": "",
                    "ood_type": "",
                    "intervention_type": intervention_type,
                    "base_endpoint": "",
                    "intervened_endpoint": "",
                    "endpoint_shift": 0.0,
                }
            )
            continue
        causal_shift = delayed_endpoint_shift(causal["base_future_frames"], causal["future_frames"])
        control_shift = delayed_endpoint_shift(control["base_future_frames"], control["future_frames"])
        causal_shifts.append(causal_shift)
        control_shifts.append(control_shift)
        records.append(
            {
                "episode_id": idx,
                "episode_type": episode["ground_truth"].get("episode_type", ""),
                "delay": episode["ground_truth"].get("delay", ""),
                "true_delayed_checkpoint_region": episode["ground_truth"].get("true_delayed_checkpoint_region", ""),
                "early_saliency_region": episode["ground_truth"].get("early_saliency_region", ""),
                "predicted_region": int(causal.get("target_region", -1)),
                "selected_delay": "",
                "correct": "",
                "ood_type": "",
                "intervention_type": intervention_type,
                "base_endpoint": endpoint_repr(causal.get("base_future_frames")),
                "intervened_endpoint": endpoint_repr(causal.get("future_frames")),
                "endpoint_shift": causal_shift,
            }
        )
    causal_drop = _mean(causal_shifts)
    control_drop = _mean(control_shifts)
    return (
        {
            "causal_trace_intervention_drop": causal_drop,
            "non_trace_stability": float(np.clip(1.0 - control_drop, 0.0, 1.0)),
            "delayed_endpoint_shift": causal_drop,
            "trace_over_control_ratio": causal_drop / (control_drop + 1e-6),
        },
        records,
    )


def endpoint_repr(frames: Any) -> str:
    if frames is None:
        return ""
    array = np.asarray(frames, dtype=np.float32)
    if array.size == 0:
        return ""
    return f"{float(array[-1].sum()):.4f}"


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0
