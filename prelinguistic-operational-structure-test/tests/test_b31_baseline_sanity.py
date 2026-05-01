from __future__ import annotations

from src.b31_baseline_sanity import run_b31_baseline_sanity_check
from src.b3_active_inspection_env import make_b3_active_inspection_episode
from src.models import make_model
from tests.conftest import small_config


def test_b31_baseline_sanity_oracle_beats_random():
    config = small_config()
    episodes = [make_b3_active_inspection_episode(config, seed + 10, "b31_baseline", delay=4) for seed in range(5)]
    models = {"recurrent_flow_checkpoint_model": make_model("recurrent_flow_checkpoint_model")}
    metrics, records = run_b31_baseline_sanity_check(models, episodes, config, seed=0)
    assert metrics["oracle_inspection_score"] >= metrics["random_inspection_score"]
    assert metrics["trace_over_saliency_gain_margin"] >= 0.0
    assert records
