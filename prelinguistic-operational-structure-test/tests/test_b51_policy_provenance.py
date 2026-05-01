from src.b51_policy_provenance import summarize_closed_loop_policy_provenance


def test_policy_provenance_summary_rates():
    records = [
        {
            "private_trace_used_for_inspect": True,
            "private_trace_used_for_update": True,
            "private_trace_used_for_intervention": True,
            "private_trace_used_for_feedback": True,
            "shared_closed_loop_policy_used": False,
            "fallback_used": False,
            "oracle_plan_used": False,
            "oracle_trace_update_used": False,
            "oracle_feedback_revision_used": False,
        }
    ]
    metrics = summarize_closed_loop_policy_provenance(records, {})
    assert metrics["private_trace_closed_loop_usage_rate"] == 1.0
    assert metrics["oracle_plan_usage_rate"] == 0.0
