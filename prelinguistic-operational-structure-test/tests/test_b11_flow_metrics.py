from __future__ import annotations

import numpy as np

from src.b11_flow_attacks import (
    endpoint_shift,
    evaluate_anti_prior_survival,
    evaluate_causal_vs_visual_deletion,
    evaluate_dynamic_decoy_rejection,
    make_anti_prior_episode,
    make_dynamic_decoy_episode,
)
from src.data import generate_episode
from src.models import make_model
from tests.conftest import small_config


def b11_small_config():
    config = small_config()
    config["b11"] = {"n_attack_episodes": 2}
    return config


def test_endpoint_shift_is_nonnegative():
    base = np.zeros((2, 64, 64, 3), dtype=np.float32)
    altered = base.copy()
    altered[-1, 20:24, 20:24, :] = 1.0
    assert endpoint_shift(base, altered) >= 0.0


def test_causal_over_visual_deletion_ratio_is_consistent():
    config = b11_small_config()
    model = make_model("flow_checkpoint_model")
    episodes = [generate_episode(config, 10, "forcefield"), generate_episode(config, 11, "collision_bounce")]
    metrics = evaluate_causal_vs_visual_deletion(model, episodes, config)
    expected = metrics["causal_deletion_shift"] / (metrics["visual_deletion_shift"] + 1e-6)
    assert metrics["causal_over_visual_deletion_ratio"] == expected
    assert metrics["causal_over_visual_deletion_ratio"] >= 0.0


def test_dynamic_decoy_rejection_returns_probability():
    config = b11_small_config()
    model = make_model("flow_checkpoint_model")
    base = generate_episode(config, 12, "forcefield")
    episodes = [make_dynamic_decoy_episode(base, config, 13)]
    metrics = evaluate_dynamic_decoy_rejection(model, episodes, config)
    assert 0.0 <= metrics["dynamic_decoy_rejection"] <= 1.0
    assert 0.0 <= metrics["decoy_selected_rate"] <= 1.0


def test_anti_prior_survival_returns_probability():
    config = b11_small_config()
    model = make_model("flow_checkpoint_model")
    episodes = [make_anti_prior_episode(config, 14, "motion_midpoint_irrelevant")]
    metrics = evaluate_anti_prior_survival(model, episodes, config)
    assert 0.0 <= metrics["anti_prior_survival"] <= 1.0
    assert 0.0 <= metrics["prior_trap_selection_rate"] <= 1.0
