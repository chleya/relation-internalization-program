from __future__ import annotations

from src.b41_shared_action_policy_ablation import evaluate_shared_action_policy_ablation
from src.b4_action_space import is_valid_action
from src.b4_intervention_env import make_b4_intervention_episode
from src.b4_intervention_policy import trace_guided_intervention_policy
from src.models import make_model
from tests.conftest import small_config


def test_shared_policy_ablation_does_not_crash_and_action_valid():
    config = small_config()
    episode = make_b4_intervention_episode(config, 7, "b41", "field")
    model = make_model("field_memory_model")
    metrics, records = evaluate_shared_action_policy_ablation(model, [episode], config)
    action = trace_guided_intervention_policy(model, episode, config)["action"]
    assert is_valid_action(action, config)
    assert metrics["shared_action_policy_ablation_drop"] >= 0.0
    assert records

