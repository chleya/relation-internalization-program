from __future__ import annotations

from src.b41_trace_ablation_specificity import evaluate_action_after_trace_ablation_specificity, matched_non_trace_ablation
from src.b4_intervention_env import make_b4_intervention_episode
from src.models import make_model
from tests.conftest import small_config


def test_trace_ablation_specificity_metrics():
    config = small_config()
    episode = make_b4_intervention_episode(config, 17, "b41", "field")
    assert matched_non_trace_ablation(episode)["b4_non_trace_ablation"] is True
    metrics, records = evaluate_action_after_trace_ablation_specificity(make_model("field_memory_model"), [episode], config)
    assert records
    assert metrics["private_trace_over_non_trace_ratio"] >= 0.0

