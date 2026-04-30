from __future__ import annotations

import pytest

from src.data import generate_episode
from src.model_io import FORBIDDEN_MODEL_INPUT_KEYS, make_model_batch
from src.models import make_model

from .conftest import small_config


def test_model_batch_excludes_future_and_ground_truth() -> None:
    episode = generate_episode(small_config(), seed=8, episode_type="forcefield")
    batch = make_model_batch(episode, small_config())
    assert FORBIDDEN_MODEL_INPUT_KEYS.isdisjoint(batch)
    assert set(batch) == {"past_frames", "future_horizon", "frame_size", "grid_size"}


def test_models_reject_unclean_episode_input() -> None:
    episode = generate_episode(small_config(), seed=9, episode_type="budgeted_inspect")
    for model_name in [
        "pixel_predictor",
        "predictive_coding_model",
        "trajectory_memory",
        "patch_graph_model",
        "koopman_model",
        "world_model",
        "slot_model",
        "field_model",
        "flow_checkpoint_model",
        "schema_model",
    ]:
        model = make_model(model_name)
        with pytest.raises(ValueError):
            model.forward(episode)
