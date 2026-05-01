from __future__ import annotations

import math

from src.b32_family_ablation import ablate_family_trace, evaluate_family_specific_trace_ablation
from src.b32_mechanism_inspection_env import make_family_specific_inspection_episode
from src.models import make_model
from tests.conftest import small_config


def test_family_trace_ablation_does_not_crash():
    config = small_config()
    episode = make_family_specific_inspection_episode(config, 31, "field_goal")
    model = make_model("field_memory_model")
    ablated = ablate_family_trace(model, episode, "field_goal", config)
    assert ablated["past_frames"].shape == episode["past_frames"].shape


def test_family_trace_ablation_metrics_are_finite():
    config = small_config()
    episodes = [make_family_specific_inspection_episode(config, 37, "schema_goal")]
    model = make_model("schema_memory_model")
    metrics, records = evaluate_family_specific_trace_ablation(model, episodes, config)
    assert records
    assert all(math.isfinite(float(value)) for value in metrics.values())
    assert "non_target_family_stability" in metrics
