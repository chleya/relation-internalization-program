from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.b22_trace_scorers import (
    compute_trace_scorer_correlation,
    score_field_trace_regions,
    score_recurrent_trace_regions,
    score_schema_trace_regions,
)
from src.model_io import make_model_batch
from src.models import make_model


def small_config():
    return {"env": {"frame_size": 64, "grid_size": 8, "past_frames": 8, "future_frames": 12}}


def test_trace_scorers_return_dict_or_not_applicable():
    config = small_config()
    batch = make_model_batch(make_delayed_checkpoint_episode(config, 0, delay=4), config)
    recurrent_scores = score_recurrent_trace_regions(make_model("recurrent_flow_checkpoint_model"), batch)
    field_scores = score_field_trace_regions(make_model("field_memory_model"), batch)
    schema_scores = score_schema_trace_regions(make_model("schema_memory_model"), batch)
    assert isinstance(recurrent_scores, dict)
    assert isinstance(field_scores, dict)
    assert isinstance(schema_scores, dict)


def test_trace_scorer_correlation_handles_missing_scorer():
    metrics = compute_trace_scorer_correlation({"recurrent": {1: 0.2, 2: 0.4}, "field": {"applicable": False}})
    assert -1.0 <= metrics["mean_trace_scorer_correlation"] <= 1.0
    assert 0.0 <= metrics["trace_family_specificity"] <= 2.0
