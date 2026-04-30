from __future__ import annotations

from src.b3_active_inspection_env import apply_inspection, compute_oracle_inspection_values, make_b3_active_inspection_episode
from tests.conftest import small_config


def b3_small_config():
    config = small_config()
    config["b3"] = {
        "frame_size": 64,
        "grid_size": 8,
        "past_frames": 8,
        "future_frames": 12,
    }
    return config


def test_b3_episode_has_inspection_metadata():
    episode = make_b3_active_inspection_episode(b3_small_config(), 0, "b3_trace_guided_inspection", delay=4)
    gt = episode["ground_truth"]
    assert "true_trace_region" in gt
    assert "saliency_region" in gt
    assert "oracle_best_inspect_region" in gt
    assert "inspection_values" in gt
    assert len(gt["inspection_values"]) == 64
    assert gt["oracle_best_inspect_region"] == gt["true_trace_region"]


def test_compute_oracle_inspection_values_marks_trace_region_best():
    episode = make_b3_active_inspection_episode(b3_small_config(), 1, "b3_trace_vs_saliency_conflict", delay=4)
    values = compute_oracle_inspection_values(episode, b3_small_config())
    true_region = episode["ground_truth"]["true_trace_region"]
    saliency_region = episode["ground_truth"]["saliency_region"]
    assert values[true_region] == 1.0
    if saliency_region != true_region:
        assert values[saliency_region] < values[true_region]


def test_apply_inspection_returns_updated_observation():
    config = b3_small_config()
    episode = make_b3_active_inspection_episode(config, 2, "b3_delay_ood_inspection", delay=5)
    region = episode["ground_truth"]["oracle_best_inspect_region"]
    updated = apply_inspection(episode, region, config)
    assert updated["inspection"]["region_id"] == region
    assert updated["inspection"]["inspected_patch"].shape[0] == 8
    assert updated["inspection"]["inspection_value"] >= 0.0
