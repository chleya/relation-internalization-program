from __future__ import annotations

from src.data import generate_episode
from src.interventions import default_interventions
from src.model_io import make_model_batch
from src.models import make_model

from .conftest import small_config


def test_interventions_do_not_crash() -> None:
    episode = generate_episode(small_config(), seed=6, episode_type="forcefield")
    model = make_model("schema_model")
    batch = make_model_batch(episode, small_config())
    for intervention in default_interventions():
        result = model.intervene_structure(batch, intervention)
        assert "applicable" in result
