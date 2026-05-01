from __future__ import annotations

from src.b41_intervention_degeneracy_audit import compute_action_type_distribution, compute_per_episode_action_overlap


def test_action_overlap_rates_are_valid_and_exact_action_uses_type_and_region():
    records = [
        {"episode_id": 0, "model": "a", "predicted_action_type": "x", "predicted_region": 1},
        {"episode_id": 0, "model": "b", "predicted_action_type": "x", "predicted_region": 1},
        {"episode_id": 0, "model": "c", "predicted_action_type": "x", "predicted_region": 2},
    ]
    metrics = compute_per_episode_action_overlap(records, {})
    assert 0.0 <= metrics["cross_model_exact_action_match_rate"] <= 1.0
    assert metrics["cross_model_exact_action_match_rate"] < 1.0
    assert metrics["same_action_type_different_region_rate"] > 0.0


def test_fixed_action_type_rate_computed():
    records = [
        {"episode_id": 0, "model": "a", "predicted_action_type": "x", "predicted_region": 1},
        {"episode_id": 1, "model": "a", "predicted_action_type": "x", "predicted_region": 2},
    ]
    metrics = compute_action_type_distribution(records, {})
    assert metrics["fixed_action_type_rate"] == 1.0

