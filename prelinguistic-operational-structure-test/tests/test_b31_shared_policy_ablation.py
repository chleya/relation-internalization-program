from __future__ import annotations

from src.b31_shared_policy_ablation import disable_shared_inspection_policy, evaluate_shared_inspection_policy_ablation
from src.b3_active_inspection_env import make_b3_active_inspection_episode
from src.models import make_model
from tests.conftest import small_config


def test_disable_shared_inspection_policy_sets_flag():
    model = make_model("field_memory_model")
    disable_shared_inspection_policy(model)
    assert getattr(model, "shared_inspection_policy_disabled") is True


def test_shared_policy_ablation_does_not_crash():
    config = small_config()
    episode = make_b3_active_inspection_episode(config, 4, "b31_shared_ablation", delay=4)
    metrics, records = evaluate_shared_inspection_policy_ablation({"field_memory_model": make_model("field_memory_model")}, [episode], config)
    assert metrics["shared_policy_ablation_drop"] >= 0.0
    assert len(records) == 1
