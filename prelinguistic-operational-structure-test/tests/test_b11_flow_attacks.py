from __future__ import annotations

import numpy as np

from src.b11_flow_attacks import (
    make_anti_prior_episode,
    make_checkpoint_relocation_ood_dataset,
    make_competing_checkpoints_episode,
    make_delayed_checkpoint_episode,
    make_dynamic_decoy_episode,
)
from src.data import generate_episode
from tests.conftest import small_config


def b11_small_config():
    config = small_config()
    config["b11"] = {
        "n_attack_episodes": 3,
        "gates": {
            "dynamic_decoy_rejection": 0.80,
            "delayed_checkpoint_accuracy": 0.75,
            "competing_checkpoint_choice": 0.75,
            "relocation_ood_stability": 0.75,
            "causal_over_visual_deletion_ratio": 1.50,
            "anti_prior_survival": 0.70,
            "causal_endpoint_shift": 0.25,
        },
    }
    return config


def test_dynamic_decoy_changes_visual_input_not_critical_region():
    config = b11_small_config()
    base = generate_episode(config, 0, "forcefield")
    critical = int(base["ground_truth"]["critical_inspection_region"])
    decoy = make_dynamic_decoy_episode(base, config, 1)
    assert np.abs(decoy["past_frames"] - base["past_frames"]).sum() > 0
    assert int(decoy["ground_truth"]["critical_inspection_region"]) == critical
    assert int(decoy["ground_truth"]["b11_decoy_region"]) != critical


def test_delayed_checkpoint_episode_has_delayed_metadata():
    config = b11_small_config()
    episode = make_delayed_checkpoint_episode(config, 2, delay=4)
    gt = episode["ground_truth"]
    assert gt["b11_attack"] == "delayed_checkpoint"
    assert gt["b11_delayed_delay"] == 4
    assert "b11_immediate_saliency_region" in gt


def test_competing_checkpoint_episode_has_multiple_candidate_regions():
    config = b11_small_config()
    episode = make_competing_checkpoints_episode(config, 3)
    candidates = episode["ground_truth"]["b11_candidate_regions"]
    assert len(candidates) >= 3
    assert episode["ground_truth"]["b11_oracle_best_region"] in candidates


def test_relocation_ood_places_checkpoint_in_rare_region():
    config = b11_small_config()
    episodes = make_checkpoint_relocation_ood_dataset(config, 4)
    assert len(episodes) == 3
    assert all(episode["ground_truth"]["b11_attack"] == "checkpoint_relocation_ood" for episode in episodes)
    assert all("b11_relocation_kind" in episode["ground_truth"] for episode in episodes)


def test_anti_prior_episode_has_trap_and_true_critical_region():
    config = b11_small_config()
    episode = make_anti_prior_episode(config, 5, "visible_occlusion_noncritical")
    gt = episode["ground_truth"]
    assert gt["b11_attack"] == "anti_prior_world"
    assert "b11_trap_region" in gt
    assert "b11_true_critical_region" in gt
    assert int(gt["b11_trap_region"]) != int(gt["b11_true_critical_region"])
