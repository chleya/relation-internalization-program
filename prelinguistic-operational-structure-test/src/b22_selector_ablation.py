from __future__ import annotations

from typing import Any

import numpy as np

from .b22_selector_free_models import make_selector_free_model
from .inspect_policy import select_region_from_logits
from .model_io import make_model_batch


def ablate_recurrent_trace(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    return ablate_trace_evidence(model, batch, "recurrent_flow_checkpoint")


def ablate_field_trace(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    return ablate_trace_evidence(model, batch, "field_memory")


def ablate_schema_trace(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    return ablate_trace_evidence(model, batch, "schema_memory")


def ablate_shared_selector(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    if str(getattr(model, "name", "")).endswith("_no_shared_selector"):
        return {"applicable": True, "output": model.forward(batch), "ablation": "already_selector_free"}
    try:
        selector_free = make_selector_free_model(str(getattr(model, "name", "")))
    except ValueError:
        return {"applicable": False, "reason": "unsupported_model"}
    return {"applicable": True, "output": selector_free.forward(batch), "ablation": "shared_selector_disabled"}


def disable_shared_selector(model: Any) -> Any:
    return make_selector_free_model(str(getattr(model, "name", "")))


def evaluate_source_specific_ablation(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    family = str(getattr(model, "structural_family", ""))
    ablator = {
        "recurrent_flow_checkpoint": ablate_recurrent_trace,
        "field_memory": ablate_field_trace,
        "schema_memory": ablate_schema_trace,
    }.get(family)
    private_drops = []
    shared_drops = []
    if ablator is None:
        return {
            "recurrent_trace_ablation_drop": 0.0,
            "field_trace_ablation_drop": 0.0,
            "schema_trace_ablation_drop": 0.0,
            "shared_selector_ablation_drop": 0.0,
            "model_private_trace_drop": 0.0,
            "shared_selector_ablation_advantage": 0.0,
        }
    for episode in episodes:
        batch = make_model_batch(episode, config)
        base_pred = select_region_from_logits(model.forward(batch).get("inspection_logits"))
        private = ablator(model, batch)
        shared = ablate_shared_selector(model, batch)
        if private.get("applicable", False):
            private_pred = select_region_from_logits(private["output"].get("inspection_logits"))
            private_drops.append(1.0 if int(private_pred) != int(base_pred) else 0.0)
        if shared.get("applicable", False):
            shared_pred = select_region_from_logits(shared["output"].get("inspection_logits"))
            shared_drops.append(1.0 if int(shared_pred) != int(base_pred) else 0.0)
    private_drop = mean_or_zero(private_drops)
    shared_drop = mean_or_zero(shared_drops)
    return {
        "recurrent_trace_ablation_drop": private_drop if family == "recurrent_flow_checkpoint" else 0.0,
        "field_trace_ablation_drop": private_drop if family == "field_memory" else 0.0,
        "schema_trace_ablation_drop": private_drop if family == "schema_memory" else 0.0,
        "shared_selector_ablation_drop": shared_drop,
        "model_private_trace_drop": private_drop,
        "shared_selector_ablation_advantage": float(shared_drop - private_drop),
    }


def evaluate_shared_selector_ablation(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    base_hits = []
    disabled_hits = []
    changed = []
    for episode in episodes:
        batch = make_model_batch(episode, config)
        true_region = int(episode["ground_truth"].get("true_delayed_checkpoint_region", episode["ground_truth"].get("critical_inspection_region", -1)))
        base_output = model.forward(batch)
        disabled = ablate_shared_selector(model, batch)
        if not disabled.get("applicable", False):
            continue
        disabled_output = disabled["output"]
        base_pred = select_region_from_logits(base_output.get("inspection_logits"))
        disabled_pred = select_region_from_logits(disabled_output.get("inspection_logits"))
        base_hits.append(1.0 if int(base_pred) == true_region else 0.0)
        disabled_hits.append(1.0 if int(disabled_pred) == true_region else 0.0)
        changed.append(1.0 if int(base_pred) != int(disabled_pred) else 0.0)
    base = mean_or_zero(base_hits)
    disabled_score = mean_or_zero(disabled_hits)
    return {
        "shared_selector_ablation_drop": max(0.0, base - disabled_score),
        "private_trace_retention_after_shared_ablation": float(disabled_score / (base + 1e-6)),
        "shared_selector_prediction_changed_rate": mean_or_zero(changed),
    }


def ablate_trace_evidence(model: Any, batch: dict[str, Any], supported_family: str) -> dict[str, Any]:
    if str(getattr(model, "structural_family", "")) != supported_family:
        return {"applicable": False, "reason": "unsupported_trace_family"}
    altered = dict(batch)
    past = np.asarray(batch["past_frames"], dtype=np.float32).copy()
    trace_mask = (past.mean(axis=-1) > 0.035) & (past.max(axis=-1) < 0.19)
    past[trace_mask] = 0.0
    altered["past_frames"] = past
    return {"applicable": True, "output": model.forward(altered), "ablation": supported_family}


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
