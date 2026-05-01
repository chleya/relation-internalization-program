from __future__ import annotations

from src.b31_inspection_provenance import collect_inspection_provenance, summarize_inspection_provenance
from src.b3_active_inspection_env import make_b3_active_inspection_episode
from src.models import make_model
from tests.conftest import small_config


def test_inspection_provenance_records_private_policy():
    config = small_config()
    episode = make_b3_active_inspection_episode(config, 0, "b31_provenance", delay=4)
    records = collect_inspection_provenance(make_model("recurrent_flow_checkpoint_model"), [episode], config)
    assert records[0]["inspect_region"] >= 0
    assert records[0]["policy_source"] == "recurrent_private_trace_inspection"
    assert records[0]["shared_inspection_policy_used"] is False
    assert records[0]["private_trace_inspection_score_used"] is True


def test_summarize_inspection_provenance_metrics():
    records = [
        {"shared_inspection_policy_used": False, "private_trace_inspection_score_used": True, "fallback_used": False, "policy_source": "x"},
        {"shared_inspection_policy_used": True, "private_trace_inspection_score_used": False, "fallback_used": True, "policy_source": "unknown"},
    ]
    metrics = summarize_inspection_provenance(records, {})
    assert metrics["shared_inspection_policy_usage_rate"] == 0.5
    assert metrics["private_trace_inspection_score_usage_rate"] == 0.5
    assert metrics["fallback_usage_rate"] == 0.5
    assert metrics["unknown_policy_source_rate"] == 0.5
