from __future__ import annotations

import math

from src.b4_intervention_env import make_b4_intervention_episode
from src.b4_trace_ablation_eval import ablate_b4_family_trace, evaluate_action_after_trace_ablation
from src.models import make_model
from tests.conftest import small_config


def test_b4_trace_ablation_returns_valid_episode():
    config = small_config()
    episode = make_b4_intervention_episode(config, 23, "b4_intervention", "field")
    model = make_model("field_memory_model")
    ablated = ablate_b4_family_trace(model, episode, "field", config)
    assert ablated["b4_ablation"]["family"] == "field"


def test_b4_trace_ablation_metrics_are_finite():
    config = small_config()
    episodes = [make_b4_intervention_episode(config, 29, "b4_intervention", "recurrent")]
    model = make_model("recurrent_flow_checkpoint_model")
    metrics, records = evaluate_action_after_trace_ablation(model, episodes, config)
    assert records
    assert all(math.isfinite(float(value)) for value in metrics.values())
    assert "trace_ablation_intervention_drop" in metrics

