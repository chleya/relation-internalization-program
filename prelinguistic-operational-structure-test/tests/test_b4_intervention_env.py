from __future__ import annotations

from src.b4_intervention_env import apply_intervention_action, evaluate_intervention_outcome, make_b4_intervention_episode
from src.b4_intervention_values import wrong_region_action_for
from tests.conftest import small_config


def test_b4_episode_has_required_intervention_fields():
    episode = make_b4_intervention_episode(small_config(), 5, "b4_intervention", "field")
    gt = episode["ground_truth"]
    assert "oracle_best_action" in gt
    assert "true_trace_region" in gt
    assert "family_best_actions" in gt
    assert set(gt["family_best_actions"]) == {"recurrent", "field", "schema"}


def test_apply_intervention_action_returns_outcome():
    config = small_config()
    episode = make_b4_intervention_episode(config, 7, "b4_intervention", "recurrent")
    intervened = apply_intervention_action(episode, episode["ground_truth"]["oracle_best_action"], config)
    outcome = evaluate_intervention_outcome(episode, intervened, config)
    assert outcome["outcome_improvement"] > 0.0
    wrong = wrong_region_action_for(episode, episode["ground_truth"]["oracle_best_action"], config)
    assert wrong["region_id"] != episode["ground_truth"]["oracle_best_action"]["region_id"]

