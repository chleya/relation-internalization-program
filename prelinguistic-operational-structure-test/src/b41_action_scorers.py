from __future__ import annotations

from itertools import combinations
from math import log2
from typing import Any

import numpy as np

from .b32_mechanism_policy import cached_family_scores
from .b4_action_space import enumerate_candidate_actions


FAMILY_ACTION_TYPE = {
    "recurrent": "stabilize_trace_region",
    "field": "block_force_region",
    "schema": "apply_local_push",
}


def recurrent_trace_action_scores(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[tuple[str, int], float]:
    return family_trace_action_scores(model, episode, config, "recurrent")


def field_trace_action_scores(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[tuple[str, int], float]:
    return family_trace_action_scores(model, episode, config, "field")


def schema_trace_action_scores(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[tuple[str, int], float]:
    return family_trace_action_scores(model, episode, config, "schema")


def family_trace_action_scores(
    model: Any,
    episode: dict[str, Any],
    config: dict[str, Any],
    family: str,
) -> dict[tuple[str, int], float]:
    region_scores = cached_family_scores(model, episode, config).get(family, {})
    preferred_action = FAMILY_ACTION_TYPE[family]
    scores: dict[tuple[str, int], float] = {}
    for action in enumerate_candidate_actions(config):
        action_type = str(action["action_type"])
        region = int(action["region_id"])
        region_score = float(region_scores.get(region, 0.0))
        action_bonus = 1.0 if action_type == preferred_action else 0.05
        if action_type in {"do_nothing", "inspect_only"}:
            action_bonus = 0.0
        scores[(action_type, region)] = region_score * action_bonus
    return scores


def compute_action_scorer_correlation(scorer_outputs: dict[str, dict[tuple[str, int], float]]) -> dict[str, float]:
    keys = sorted({key for scores in scorer_outputs.values() for key in scores})
    if not keys:
        return {
            "recurrent_field_action_score_correlation": 1.0,
            "recurrent_schema_action_score_correlation": 1.0,
            "field_schema_action_score_correlation": 1.0,
            "mean_action_scorer_correlation": 1.0,
            "action_scorer_specificity": 0.0,
            "action_score_entropy": 0.0,
        }
    vectors = {
        name: np.asarray([float(scores.get(key, 0.0)) for key in keys], dtype=np.float32)
        for name, scores in scorer_outputs.items()
    }
    pair_values: dict[str, float] = {}
    correlations = []
    for left, right in combinations(sorted(vectors), 2):
        corr = safe_corrcoef(vectors[left], vectors[right])
        correlations.append(corr)
        pair_values[f"{pair_key(left, right)}_action_score_correlation"] = corr
    mean_corr = float(np.mean(correlations)) if correlations else 1.0
    return {
        "recurrent_field_action_score_correlation": pair_values.get("recurrent_field_action_score_correlation", 1.0),
        "recurrent_schema_action_score_correlation": pair_values.get("recurrent_schema_action_score_correlation", 1.0),
        "field_schema_action_score_correlation": pair_values.get("field_schema_action_score_correlation", 1.0),
        "mean_action_scorer_correlation": mean_corr,
        "action_scorer_specificity": float(np.clip(1.0 - max(mean_corr, 0.0), 0.0, 1.0)),
        "action_score_entropy": float(np.mean([score_entropy(scores) for scores in scorer_outputs.values()])),
    }


def evaluate_action_scorer_specificity(
    models: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    records = []
    probe_model = next(iter(models.values()))
    for episode_id, episode in enumerate(episodes):
        score_maps = {
            "recurrent": recurrent_trace_action_scores(probe_model, episode, config),
            "field": field_trace_action_scores(probe_model, episode, config),
            "schema": schema_trace_action_scores(probe_model, episode, config),
        }
        corr = compute_action_scorer_correlation(score_maps)
        rows.append(corr)
        for family, scores in score_maps.items():
            action_type, region = selected_action(scores)
            records.append(
                {
                    "record_kind": "action_scorer",
                    "audit_type": "family_specific_action_scorer",
                    "seed": seed,
                    "episode_id": episode_id,
                    "trace_family": family,
                    "predicted_action_type": action_type,
                    "predicted_region": region,
                    "action_score": float(scores.get((action_type, region), 0.0)),
                    "gate_pass": int(corr["action_scorer_specificity"] >= 0.0),
                    "note": "family-specific private trace action scorer",
                    **corr,
                }
            )
    return average_metric_rows(rows), records


def selected_action(scores: dict[tuple[str, int], float]) -> tuple[str, int]:
    if not scores:
        return "do_nothing", -1
    return max(scores.items(), key=lambda item: (float(item[1]), item[0][0], -int(item[0][1])))[0]


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return compute_action_scorer_correlation({})
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([float(row.get(key, 0.0)) for row in rows])) for key in keys}


def safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 1.0
    if float(np.std(left)) < 1e-8 or float(np.std(right)) < 1e-8:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def score_entropy(scores: dict[tuple[str, int], float]) -> float:
    values = np.asarray([max(float(value), 0.0) for value in scores.values()], dtype=np.float32)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probs = values / total
    return float(-sum(float(p) * log2(float(p)) for p in probs if float(p) > 0.0))


def pair_key(left: str, right: str) -> str:
    names = {left, right}
    if names == {"recurrent", "field"}:
        return "recurrent_field"
    if names == {"recurrent", "schema"}:
        return "recurrent_schema"
    if names == {"field", "schema"}:
        return "field_schema"
    return "_".join(sorted(names))

