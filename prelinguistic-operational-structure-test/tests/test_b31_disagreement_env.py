from __future__ import annotations

from src.b31_disagreement_inspection_env import evaluate_disagreement_inspection_divergence, make_disagreement_inspection_episode
from src.models import make_model
from tests.conftest import small_config


def test_disagreement_inspection_episode_has_distinct_regions():
    episode = make_disagreement_inspection_episode(small_config(), 2, "schema_memory")
    gt = episode["ground_truth"]
    regions = {gt["recurrent_trace_region"], gt["field_trace_region"], gt["schema_trace_region"]}
    assert len(regions) == 3
    assert gt["causal_family"] == "schema_memory"
    assert gt["oracle_best_inspect_region"] == gt["schema_trace_region"]


def test_disagreement_inspection_divergence_runs():
    config = small_config()
    episodes = [make_disagreement_inspection_episode(config, 3, "recurrent_memory")]
    models = {
        "recurrent_flow_checkpoint_model": make_model("recurrent_flow_checkpoint_model"),
        "field_memory_model": make_model("field_memory_model"),
        "schema_memory_model": make_model("schema_memory_model"),
    }
    metrics, records = evaluate_disagreement_inspection_divergence(models, episodes, config)
    assert 0.0 <= metrics["disagreement_inspection_divergence"] <= 1.0
    assert len(records) == 3
