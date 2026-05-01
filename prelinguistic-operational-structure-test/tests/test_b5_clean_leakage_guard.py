import pytest

from src.b5_clean_leakage_guard import guard_baseline_access, guard_policy_input, guard_policy_output


def test_guard_policy_input_detects_forbidden_key():
    report = guard_policy_input({"nested": {"oracle_value": 1.0}}, {"b5_clean": {"fail_fast_on_leakage": False}})
    assert report["model_input_leakage_count"] == 1
    assert "nested.oracle_value" in report["forbidden_key_paths"]


def test_guard_policy_input_fail_fast_raises():
    with pytest.raises(AssertionError):
        guard_policy_input({"ground_truth": {}}, {"b5_clean": {"fail_fast_on_leakage": True}})


def test_guard_policy_output_detects_oracle_usage_flag():
    report = guard_policy_output({"provenance": {"oracle_plan_used": True}}, {})
    assert report["policy_output_oracle_usage_rate"] == 1.0


def test_non_oracle_baseline_cannot_access_oracle_view():
    report = guard_baseline_access("random_closed_loop_baseline", {"oracle_baseline_view": {}}, {"b5_clean": {"fail_fast_on_leakage": False}})
    assert report["oracle_baseline_access_violation_count"] == 1


def test_oracle_baseline_can_access_oracle_view():
    report = guard_baseline_access("oracle_closed_loop_baseline", {"oracle_baseline_view": {}}, {})
    assert report["oracle_baseline_access_violation_count"] == 0
