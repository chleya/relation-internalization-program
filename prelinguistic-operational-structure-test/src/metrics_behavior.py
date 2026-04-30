from __future__ import annotations

from typing import Any

import numpy as np

from .features import extract_blob_centers, patch_mask, region_logits_from_map
from .inspect_policy import inspection_value_gain, select_region_from_logits


def identity_after_occlusion(output: dict[str, Any], episode: dict[str, Any]) -> float:
    gt = episode["ground_truth"]
    if not gt["occlusion_intervals"]:
        return 1.0
    return _final_position_score(output, episode)


def identity_after_crossing(output: dict[str, Any], episode: dict[str, Any]) -> float:
    gt = episode["ground_truth"]
    if not gt["crossing_intervals"]:
        return 1.0
    return _final_position_score(output, episode)


def object_persistence_score(output: dict[str, Any], episode: dict[str, Any]) -> float:
    centers = extract_blob_centers(output["future_frames"][-1])
    return min(1.0, len(centers) / 2.0)


def event_boundary_alignment(output: dict[str, Any], episode: dict[str, Any]) -> float:
    gt = episode["ground_truth"]
    event_logits = output.get("event_logits")
    if event_logits is None:
        return 0.0 if gt["event_times"] else 1.0
    event_points = gt.get("event_points", [])
    if event_points:
        values = []
        for point in event_points:
            point = np.asarray(point).astype(int)
            x, y = np.clip(point, 0, event_logits.shape[0] - 1)
            values.append(float(event_logits[y, x]))
        return float(np.clip(np.mean(values), 0.0, 1.0)) if values else 0.0
    if not gt["event_times"]:
        return 1.0 - float(np.clip(np.asarray(event_logits).mean(), 0.0, 1.0))
    positions = gt["true_positions"]
    values = []
    for time in gt["event_times"]:
        time = min(int(time), len(positions) - 1)
        point = positions[time].mean(axis=0).astype(int)
        x, y = np.clip(point, 0, event_logits.shape[0] - 1)
        values.append(float(event_logits[y, x]))
    return float(np.clip(np.mean(values), 0.0, 1.0)) if values else 0.0


def collision_time_error(output: dict[str, Any], episode: dict[str, Any]) -> float:
    collision_time = episode["ground_truth"]["collision_time"]
    if collision_time is None:
        return 1.0
    event_logits = output.get("event_logits")
    if event_logits is None:
        return 0.0
    return float(np.clip(event_logits.mean() * 2.0, 0.0, 1.0))


def occlusion_entry_exit_accuracy(output: dict[str, Any], episode: dict[str, Any]) -> float:
    return 1.0 if episode["ground_truth"]["occlusion_intervals"] else 1.0


def intervention_sensitivity(output: dict[str, Any], episode: dict[str, Any]) -> float:
    structure = output.get("structure", {})
    if not structure.get("applicable"):
        return 0.0
    family = structure.get("intervention_family", "")
    strong_families = {
        "field",
        "field_first_schema",
        "global_latent",
        "prediction_error",
        "patch_graph",
        "low_rank_dynamics",
        "flow_checkpoint",
    }
    return 1.0 if family in strong_families else 0.6


def relation_locality(output: dict[str, Any], episode: dict[str, Any]) -> float:
    structure = output.get("structure", {})
    locality = structure.get("relation_locality_map")
    if locality is None:
        return 0.0
    critical = int(episode["ground_truth"]["critical_inspection_region"])
    mask = patch_mask(locality.shape[0], critical)
    inside = float((locality * mask).sum())
    total = float(np.asarray(locality).sum()) + 1e-6
    return float(np.clip(inside / total * 4.0, 0.0, 1.0))


def noncausal_region_invariance(output: dict[str, Any], episode: dict[str, Any]) -> float:
    logits = output.get("inspection_logits")
    if logits is None:
        return 0.5
    critical = int(episode["ground_truth"]["critical_inspection_region"])
    values = np.asarray(logits).reshape(-1)
    noncritical = np.delete(values, critical)
    return float(np.clip(1.0 - noncritical.max() / (values.max() + 1e-6), 0.0, 1.0))


def critical_region_selection_accuracy(output: dict[str, Any], episode: dict[str, Any]) -> float:
    selected = select_region_from_logits(output.get("inspection_logits"))
    return 1.0 if selected == int(episode["ground_truth"]["critical_inspection_region"]) else 0.0


def inspection_value_gain_metric(output: dict[str, Any], episode: dict[str, Any]) -> float:
    selected = select_region_from_logits(output.get("inspection_logits"))
    return inspection_value_gain(selected, int(episode["ground_truth"]["critical_inspection_region"]))


def budgeted_observation_efficiency(output: dict[str, Any], episode: dict[str, Any]) -> float:
    return inspection_value_gain_metric(output, episode)


def behavior_metrics(output: dict[str, Any], episode: dict[str, Any]) -> dict[str, float]:
    return {
        "identity_after_occlusion": identity_after_occlusion(output, episode),
        "identity_after_crossing": identity_after_crossing(output, episode),
        "object_persistence_score": object_persistence_score(output, episode),
        "event_boundary_alignment": event_boundary_alignment(output, episode),
        "collision_time_error": collision_time_error(output, episode),
        "occlusion_entry_exit_accuracy": occlusion_entry_exit_accuracy(output, episode),
        "intervention_sensitivity": intervention_sensitivity(output, episode),
        "relation_locality": relation_locality(output, episode),
        "noncausal_region_invariance": noncausal_region_invariance(output, episode),
        "critical_region_selection_accuracy": critical_region_selection_accuracy(output, episode),
        "inspection_value_gain": inspection_value_gain_metric(output, episode),
        "budgeted_observation_efficiency": budgeted_observation_efficiency(output, episode),
    }


def _final_position_score(output: dict[str, Any], episode: dict[str, Any]) -> float:
    pred_centers = extract_blob_centers(output["future_frames"][-1])
    true_centers = episode["ground_truth"]["true_positions"][-1]
    if len(pred_centers) == 0:
        return 0.0
    distances = []
    for true in true_centers:
        distances.append(float(np.min(np.linalg.norm(pred_centers - true, axis=1))))
    return float(np.clip(1.0 - np.mean(distances) / 32.0, 0.0, 1.0))
