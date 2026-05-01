from __future__ import annotations

from typing import Any

import numpy as np

from .b5_feedback_revision import feedback_revision_accuracy, observe_consequence, revise_trace_after_feedback
from .b5_trace_update import export_trace_state


def evaluate_feedback_revision_specificity(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], seed: int = 0, model_name: str = "") -> tuple[dict[str, float], list[dict[str, Any]]]:
    base_hits = []
    ablated_hits = []
    scripted_hits = []
    shuffled_hits = []
    contradictory_hits = []
    records = []
    for episode_id, episode in enumerate(episodes):
        trace_state = export_trace_state(model, episode, config)
        action = episode["ground_truth"]["oracle_intervention_action"]
        consequence = observe_consequence(episode, action, config)
        base = revise_trace_after_feedback(trace_state, consequence, episode, config)
        ablated = dict(trace_state)
        scripted = scripted_feedback_revision_baseline(episode, consequence, config)
        shuffled_episode = episodes[(episode_id + 1) % len(episodes)]
        shuffled = revise_trace_after_feedback(trace_state, observe_consequence(shuffled_episode, shuffled_episode["ground_truth"]["oracle_intervention_action"], config), episode, config)
        contradictory = revise_trace_after_feedback(trace_state, {"consequence_value": 0.0, "observed_trace_region": int(episode["ground_truth"]["wrong_inspect_region"]), "action_success": False}, episode, config)
        base_hit = feedback_revision_accuracy(base, episode)
        ablated_hit = feedback_revision_accuracy(ablated, episode)
        scripted_hit = feedback_revision_accuracy(scripted, episode)
        shuffled_hit = feedback_revision_accuracy(shuffled, episode)
        contradictory_hit = feedback_revision_accuracy(contradictory, episode)
        base_hits.append(base_hit)
        ablated_hits.append(ablated_hit)
        scripted_hits.append(scripted_hit)
        shuffled_hits.append(shuffled_hit)
        contradictory_hits.append(contradictory_hit)
        records.append(
            {
                "record_kind": "feedback_revision_specificity",
                "model": model_name,
                "episode_id": episode_id,
                "feedback_revision_source": "private_trace_feedback_revision",
                "trace_after_feedback_region": int(base["region"]),
                "gate_pass": int(base_hit),
                "note": "feedback revision specificity audit",
            }
        )
    base_score = mean_or_zero(base_hits)
    ablated_score = mean_or_zero(ablated_hits)
    scripted_score = mean_or_zero(scripted_hits)
    return {
        "feedback_revision_specificity": base_score,
        "feedback_revision_ablation_drop": max(0.0, base_score - ablated_score),
        "feedback_revision_over_scripted_ratio": base_score / (scripted_score + 1e-6),
        "shuffled_feedback_revision_drop": max(0.0, base_score - mean_or_zero(shuffled_hits)),
        "contradictory_feedback_sensitivity": max(0.0, base_score - mean_or_zero(contradictory_hits)),
        "future_decision_improvement_after_feedback": base_score,
    }, records


def scripted_feedback_revision_baseline(episode: dict[str, Any], consequence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "region": int(episode["ground_truth"]["true_trace_region"]),
        "uncertainty": 0.05,
        "confidence": 0.95,
        "source": "scripted_feedback",
    }


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
