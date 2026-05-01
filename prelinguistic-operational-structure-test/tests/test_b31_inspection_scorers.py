from __future__ import annotations

from src.b31_inspection_scorers import (
    compute_inspection_scorer_correlation,
    field_trace_inspection_scores,
    recurrent_trace_inspection_scores,
    schema_trace_inspection_scores,
)
from src.b31_disagreement_inspection_env import make_disagreement_inspection_episode
from src.models import make_model
from tests.conftest import small_config


def test_trace_family_inspection_scorers_return_region_scores():
    config = small_config()
    episode = make_disagreement_inspection_episode(config, 1, "field_trace")
    scorers = [
        recurrent_trace_inspection_scores(make_model("recurrent_flow_checkpoint_model"), episode, config),
        field_trace_inspection_scores(make_model("field_memory_model"), episode, config),
        schema_trace_inspection_scores(make_model("schema_memory_model"), episode, config),
    ]
    assert all(scores for scores in scorers)
    assert all(all(0 <= int(region) < 64 for region in scores) for scores in scorers)


def test_compute_inspection_scorer_correlation_valid_range():
    metrics = compute_inspection_scorer_correlation(
        {
            "recurrent": {1: 1.0, 2: 0.0},
            "field": {1: 0.0, 2: 1.0},
            "schema": {1: 0.5, 2: 0.5},
        }
    )
    assert -1.0 <= metrics["mean_inspection_scorer_correlation"] <= 1.0
    assert 0.0 <= metrics["inspection_scorer_specificity"] <= 1.0
