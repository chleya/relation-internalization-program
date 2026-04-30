from __future__ import annotations

from src.data import generate_episode
from src.metrics_behavior import behavior_metrics
from src.model_io import make_model_batch
from src.models import make_model

from .conftest import small_config


def test_behavior_metrics_return_expected_keys() -> None:
    episode = generate_episode(small_config(), seed=5, episode_type="budgeted_inspect")
    model = make_model("schema_model")
    output = model.forward(make_model_batch(episode, small_config()))
    metrics = behavior_metrics(output, episode)
    assert "identity_after_occlusion" in metrics
    assert "event_boundary_alignment" in metrics
    assert "critical_region_selection_accuracy" in metrics
