from __future__ import annotations

from typing import Any

import numpy as np

from .b5_closed_loop_env import inspect_region
from .b5_trace_update import export_trace_state, trace_update_accuracy, update_trace_after_inspection


def evaluate_trace_update_specificity(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], seed: int = 0, model_name: str = "") -> tuple[dict[str, float], list[dict[str, Any]]]:
    base_hits = []
    ablated_hits = []
    non_trace_hits = []
    wrong_hits = []
    shuffled_hits = []
    scripted_hits = []
    records = []
    for episode_id, episode in enumerate(episodes):
        trace_state = export_trace_state(model, episode, config)
        correct_obs = inspect_region(episode, int(episode["ground_truth"]["oracle_inspect_region"]), config)
        wrong_obs = inspect_region(episode, int(episode["ground_truth"]["wrong_inspect_region"]), config)
        shuffled_episode = episodes[(episode_id + 1) % len(episodes)]
        shuffled_obs = inspect_region(shuffled_episode, int(shuffled_episode["ground_truth"]["oracle_inspect_region"]), config)
        base = update_trace_after_inspection(trace_state, correct_obs, episode, config)
        ablated = ablate_trace_update_path(model, trace_state, correct_obs, config)
        non_trace = dict(trace_state)
        non_trace["region"] = int(correct_obs["observed_trace_region"])
        wrong = update_trace_after_inspection(trace_state, wrong_obs, episode, config)
        shuffled = update_trace_after_inspection(trace_state, shuffled_obs, episode, config)
        scripted = scripted_trace_update_baseline(episode, correct_obs, config)
        base_hit = trace_update_accuracy(base, episode)
        ablated_hit = trace_update_accuracy(ablated, episode)
        non_trace_hit = trace_update_accuracy(non_trace, episode)
        wrong_hit = trace_update_accuracy(wrong, episode)
        shuffled_hit = trace_update_accuracy(shuffled, episode)
        scripted_hit = trace_update_accuracy(scripted, episode)
        base_hits.append(base_hit)
        ablated_hits.append(ablated_hit)
        non_trace_hits.append(non_trace_hit)
        wrong_hits.append(wrong_hit)
        shuffled_hits.append(shuffled_hit)
        scripted_hits.append(scripted_hit)
        records.append(
            {
                "record_kind": "trace_update_specificity",
                "model": model_name,
                "episode_id": episode_id,
                "trace_update_source": "private_trace_update",
                "trace_before_region": int(trace_state["region"]),
                "trace_after_inspection_region": int(base["region"]),
                "gate_pass": int(base_hit),
                "note": "trace update specificity audit",
            }
        )
    base_score = mean_or_zero(base_hits)
    ablated_score = mean_or_zero(ablated_hits)
    non_trace_score = mean_or_zero(non_trace_hits)
    scripted_score = mean_or_zero(scripted_hits)
    ablation_drop = max(0.0, base_score - ablated_score)
    non_trace_drop = max(0.0, base_score - non_trace_score)
    return {
        "trace_update_specificity": base_score,
        "trace_update_ablation_drop": ablation_drop,
        "trace_update_over_non_trace_ratio": ablation_drop / (non_trace_drop + 1e-6),
        "wrong_inspection_update_drop": max(0.0, base_score - mean_or_zero(wrong_hits)),
        "shuffled_inspection_update_drop": max(0.0, base_score - mean_or_zero(shuffled_hits)),
        "scripted_update_score": scripted_score,
        "model_gain_over_scripted_update": base_score - scripted_score,
    }, records


def ablate_trace_update_path(model: Any, trace_state: dict[str, Any], inspection_observation: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(trace_state)
    updated["source"] = "trace_update_ablation"
    return updated


def scripted_trace_update_baseline(episode: dict[str, Any], inspection_observation: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "region": int(inspection_observation.get("observed_trace_region", episode["ground_truth"]["initial_trace_region"])),
        "uncertainty": 0.05,
        "confidence": 0.95,
        "source": "scripted_update",
    }


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
