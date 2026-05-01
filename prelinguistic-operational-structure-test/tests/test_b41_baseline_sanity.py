from __future__ import annotations

from src.b41_baseline_sanity import run_b41_baseline_sanity_check
from src.b4_intervention_env import make_b4_intervention_episode
from tests.conftest import small_config


def test_baseline_sanity_runs_and_oracle_beats_random():
    config = small_config()
    episode = make_b4_intervention_episode(config, 13, "b41", "recurrent")
    metrics, records = run_b41_baseline_sanity_check([episode], config, seed=0)
    assert records
    assert metrics["oracle_intervention_score"] >= metrics["random_intervention_score"]
    assert "inspect_only_score" in metrics

