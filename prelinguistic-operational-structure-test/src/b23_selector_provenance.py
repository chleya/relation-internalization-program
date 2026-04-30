from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .inspect_policy import select_region_from_logits
from .model_io import make_model_batch


VALID_FAMILIES = {"recurrent_memory", "field_trace", "schema_memory"}


def collect_b23_provenance(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    model_name = str(getattr(model, "name", model.__class__.__name__))
    for idx, episode in enumerate(episodes):
        batch = make_model_batch(episode, config)
        output = model.forward(batch)
        structure = output.get("structure", {})
        selected = int(structure.get("selected_region", select_region_from_logits(output.get("inspection_logits"))))
        scores = structure.get("private_trace_scores", {})
        rows.append(
            {
                "model": model_name,
                "episode_id": idx,
                "selected_region": selected,
                "source_trace_family": str(structure.get("source_trace_family", "unknown")),
                "source_module": str(structure.get("source_module", "unknown")),
                "shared_selector_used": bool(structure.get("shared_selector_used", False)),
                "fallback_used": bool(structure.get("fallback_used", False)),
                "model_private_score_used": bool(structure.get("model_private_score_used", False)),
                "private_trace_score": float(scores.get(selected, structure.get("trace_confidence", 0.0))) if isinstance(scores, dict) else 0.0,
                "candidate_scores": serialize_scores(scores),
            }
        )
    return rows


def summarize_b23_provenance(records: list[dict[str, Any]]) -> dict[str, float]:
    total = len(records) or 1
    shared = sum(1 for row in records if bool(row.get("shared_selector_used")))
    fallback = sum(1 for row in records if bool(row.get("fallback_used")))
    private = sum(1 for row in records if bool(row.get("model_private_score_used")))
    family_counts = Counter(str(row.get("source_trace_family", "unknown")) for row in records)
    valid_family = sum(count for family, count in family_counts.items() if family in VALID_FAMILIES)
    return {
        "shared_selector_usage_rate": float(shared / total),
        "fallback_usage_rate": float(fallback / total),
        "model_private_score_usage_rate": float(private / total),
        "valid_trace_family_rate": float(valid_family / total),
    }


def serialize_scores(scores: Any) -> str:
    if not isinstance(scores, dict):
        return ""
    return ";".join(f"{int(region)}:{float(score):.4f}" for region, score in sorted(scores.items(), key=lambda item: int(item[0])))


def prediction_overlap_by_episode(records: list[dict[str, Any]]) -> float:
    grouped: dict[int, list[int]] = {}
    for row in records:
        grouped.setdefault(int(row.get("episode_id", -1)), []).append(int(row.get("selected_region", -1)))
    values = [1.0 if len(set(regions)) <= 1 and len(regions) > 1 else 0.0 for regions in grouped.values()]
    return float(np.mean(values)) if values else 0.0
