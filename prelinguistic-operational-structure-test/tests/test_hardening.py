from __future__ import annotations

from src.hardening import (
    add_decoy_patch,
    causal_endpoint_shift,
    decoy_patch_stability,
    make_static_decoy_batch,
    static_decoy_suppression,
)
from src.data import generate_dataset, generate_episode
from src.model_io import make_model_batch
from src.models import make_model

from .conftest import small_config


def test_decoy_patch_changes_pixels_but_keeps_metadata() -> None:
    episode = generate_episode(small_config(), seed=21, episode_type="forcefield")
    decoy = add_decoy_patch(episode, small_config())
    assert decoy["ground_truth"]["critical_inspection_region"] == episode["ground_truth"]["critical_inspection_region"]
    assert float(decoy["past_frames"].sum()) > float(episode["past_frames"].sum())


def test_static_decoy_batch_is_suppressed() -> None:
    model = make_model("flow_checkpoint_model")
    assert static_decoy_suppression(model, small_config(), seed=22) == 1.0
    output = model.forward(make_static_decoy_batch(small_config(), seed=22))
    assert output["structure"]["applicable"] is False


def test_flow_checkpoint_hardening_metrics_are_computable() -> None:
    model = make_model("flow_checkpoint_model")
    dataset = generate_dataset(small_config(), "test", seed=23)
    assert 0.0 <= decoy_patch_stability(model, dataset, small_config()) <= 1.0
    assert 0.0 <= causal_endpoint_shift(model, dataset, small_config()) <= 1.0
    batch = make_model_batch(dataset[0], small_config())
    result = model.intervene_structure(batch, {"type": "event_latent_perturbation"})
    assert "applicable" in result
