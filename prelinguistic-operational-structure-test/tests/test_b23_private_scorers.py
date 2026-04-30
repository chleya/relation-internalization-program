from src.b2_delayed_env import make_delayed_checkpoint_episode
from src.b23_private_scorers import compute_private_scorer_correlation, private_trace_scores
from src.model_io import make_model_batch


def small_config():
    return {"env": {"frame_size": 64, "grid_size": 8, "past_frames": 8, "future_frames": 12}}


def test_private_trace_scores_return_valid_regions():
    config = small_config()
    batch = make_model_batch(make_delayed_checkpoint_episode(config, 0, delay=4), config)
    scores = private_trace_scores(batch, "field_memory")
    assert scores
    assert all(0 <= int(region) < 64 for region in scores)


def test_private_scorer_correlation_valid_range():
    metrics = compute_private_scorer_correlation(
        {
            "recurrent": {1: 1.0, 2: 0.2, 3: 0.1},
            "field": {1: 0.2, 2: 1.0, 3: 0.1},
            "schema": {1: 0.1, 2: 0.2, 3: 1.0},
        }
    )
    assert 0.0 <= metrics["trace_family_specificity"] <= 1.0
    assert -1.0 <= metrics["mean_trace_scorer_correlation"] <= 1.0
