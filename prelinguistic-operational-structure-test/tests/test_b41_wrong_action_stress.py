from __future__ import annotations

from src.b41_wrong_action_stress import construct_wrong_action_variants, evaluate_wrong_action_wrong_region_stress
from src.b4_intervention_env import make_b4_intervention_episode
from src.b4_intervention_values import compute_intervention_value
from tests.conftest import small_config


def test_wrong_action_variants_and_penalties():
    config = small_config()
    episode = make_b4_intervention_episode(config, 11, "b41", "schema")
    correct = episode["ground_truth"]["oracle_best_action"]
    variants = construct_wrong_action_variants(episode, correct, config)
    assert variants["wrong_action"]["action_type"] != correct["action_type"]
    assert variants["wrong_region"]["region_id"] != correct["region_id"]
    metrics, _ = evaluate_wrong_action_wrong_region_stress([episode], config)
    assert metrics["wrong_action_penalty"] >= 0.0
    assert compute_intervention_value(episode, correct, config) >= compute_intervention_value(episode, variants["wrong_action"], config)

