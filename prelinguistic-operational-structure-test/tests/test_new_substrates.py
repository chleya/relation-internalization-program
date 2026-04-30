from __future__ import annotations

from src.data import generate_episode
from src.model_io import make_model_batch
from src.models import make_model

from .conftest import small_config


def test_new_substrate_models_forward_without_labels() -> None:
    episode = generate_episode(small_config(), seed=12, episode_type="forcefield")
    batch = make_model_batch(episode, small_config())
    for model_name in ["predictive_coding_model", "patch_graph_model", "koopman_model", "flow_checkpoint_model"]:
        model = make_model(model_name)
        model.fit([batch], small_config())
        output = model.forward(batch)
        assert output["future_frames"].shape[0] == batch["future_horizon"]
        assert output["structure"]["applicable"] is True
        assert output["inspection_logits"] is not None
