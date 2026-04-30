from __future__ import annotations

from collections import Counter
from math import log2
from typing import Any

import numpy as np

from .inspect_policy import select_region_from_logits
from .model_io import make_model_batch


BASE_SHARED_MODELS = {
    "recurrent_flow_checkpoint_model",
    "field_memory_model",
    "schema_memory_model",
}

PRIVATE_FAMILY = {
    "recurrent_flow_checkpoint": "recurrent_memory",
    "field_memory": "field_trace",
    "schema_memory": "schema_memory",
}


def collect_trace_provenance(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    model_name = str(getattr(model, "name", model.__class__.__name__))
    for idx, episode in enumerate(episodes):
        batch = make_model_batch(episode, config)
        output = model.forward(batch)
        structure = output.get("structure", {}) or {}
        selected_region = int(structure.get("selected_region", select_region_from_logits(output.get("inspection_logits"))))
        shared_selector_used = bool(structure.get("shared_selector_used", model_name in BASE_SHARED_MODELS))
        model_private_score_used = bool(structure.get("model_private_score_used", False))
        fallback_used = bool(structure.get("fallback_used", False))
        source_module = str(structure.get("source_module", ""))
        source_family = str(structure.get("source_trace_family", ""))
        if shared_selector_used:
            source_module = source_module or "delayed_common.select_delayed_region"
            source_family = "shared_selector"
        elif model_private_score_used:
            source_module = source_module or f"{model_name}.private_trace_scorer"
            source_family = source_family or PRIVATE_FAMILY.get(str(structure.get("trace_family", "")), "unknown")
        elif fallback_used:
            source_module = source_module or "fallback"
            source_family = "fallback"
        else:
            source_module = source_module or "unknown"
            source_family = source_family or "unknown"
        records.append(
            {
                "model": model_name,
                "episode_id": idx,
                "selected_region": selected_region,
                "source_module": source_module,
                "source_trace_family": source_family,
                "source_score": float(structure.get("trace_confidence", 0.0)),
                "fallback_used": bool(fallback_used),
                "shared_selector_used": bool(shared_selector_used),
                "model_private_score_used": bool(model_private_score_used),
                "candidate_scores": serialize_scores(structure.get("candidate_scores", {})),
                "true_region": int(episode.get("ground_truth", {}).get("true_delayed_checkpoint_region", -1)),
            }
        )
    return records


def summarize_trace_provenance(provenance_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    total = len(provenance_records) or 1
    shared = sum(1 for row in provenance_records if bool(row.get("shared_selector_used")))
    private = sum(1 for row in provenance_records if bool(row.get("model_private_score_used")))
    fallback = sum(1 for row in provenance_records if bool(row.get("fallback_used")))
    unknown = sum(1 for row in provenance_records if str(row.get("source_trace_family", "")) in {"", "unknown"})
    family_counts = Counter(str(row.get("source_trace_family", "unknown")) for row in provenance_records)
    return {
        "shared_selector_usage_rate": float(shared / total),
        "model_private_score_usage_rate": float(private / total),
        "fallback_usage_rate": float(fallback / total),
        "unknown_source_rate": float(unknown / total),
        "source_family_entropy": entropy(family_counts),
    }


def serialize_scores(scores: Any) -> str:
    if not isinstance(scores, dict):
        return ""
    return ";".join(f"{int(region)}:{float(score):.4f}" for region, score in sorted(scores.items(), key=lambda item: int(item[0])))


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            value -= p * log2(p)
    return float(value)


def cross_model_prediction_match_rate(provenance_records: list[dict[str, Any]]) -> float:
    grouped: dict[int, list[int]] = {}
    for row in provenance_records:
        grouped.setdefault(int(row.get("episode_id", -1)), []).append(int(row.get("selected_region", -1)))
    values = [1.0 if len(set(regions)) <= 1 and len(regions) > 1 else 0.0 for regions in grouped.values()]
    return float(np.mean(values)) if values else 0.0
