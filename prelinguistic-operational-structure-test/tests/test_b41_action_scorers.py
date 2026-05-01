from __future__ import annotations

import math

from src.b41_action_scorers import compute_action_scorer_correlation, recurrent_trace_action_scores
from src.b4_intervention_env import make_b4_intervention_episode
from src.models import make_model
from tests.conftest import small_config


def test_action_scorer_returns_finite_tuple_scores():
    config = small_config()
    episode = make_b4_intervention_episode(config, 5, "b41", "recurrent")
    scores = recurrent_trace_action_scores(make_model("recurrent_flow_checkpoint_model"), episode, config)
    assert scores
    key = next(iter(scores))
    assert isinstance(key, tuple)
    assert math.isfinite(float(scores[key]))


def test_action_scorer_correlation_valid_for_missing_scorer():
    metrics = compute_action_scorer_correlation({"recurrent": {("a", 1): 1.0}, "field": {}, "schema": {}})
    assert 0.0 <= metrics["action_scorer_specificity"] <= 1.0

