from __future__ import annotations

import numpy as np

from src.b3_active_inspection_env import make_b3_active_inspection_episode
from src.b3_trace_ablation_eval import ablate_episode_region, evaluate_inspection_after_trace_ablation
from src.models import make_model
from tests.conftest import small_config


def test_ablate_episode_region_changes_past_frames():
    config = small_config()
    episode = make_b3_active_inspection_episode(config, 6, "b3_trace_guided_inspection", delay=4)
    region = episode["ground_truth"]["true_trace_region"]
    ablated = ablate_episode_region(episode, region, config)
    assert np.sum(np.abs(episode["past_frames"] - ablated["past_frames"])) > 0.0


def test_inspection_after_trace_ablation_returns_finite_metrics():
    config = small_config()
    episodes = [make_b3_active_inspection_episode(config, seed + 20, "b3_trace_guided_inspection", delay=4) for seed in range(3)]
    metrics, records = evaluate_inspection_after_trace_ablation(make_model("schema_memory_model"), episodes, config)
    assert 0.0 <= metrics["trace_ablated_inspection_accuracy"] <= 1.0
    assert metrics["trace_ablation_inspection_drop"] >= 0.0
    assert len(records) == len(episodes)
