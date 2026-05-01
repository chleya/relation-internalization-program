from src.b5_clean_leakage_guard import guard_baseline_access


def test_clean_oracle_access_rules():
    oracle = guard_baseline_access("oracle_closed_loop_baseline", {"oracle_baseline_view": {"oracle_closed_loop_plan": {}}}, {})
    random = guard_baseline_access(
        "random_closed_loop_baseline",
        {"model_input": {}, "oracle_baseline_view": {"oracle_closed_loop_plan": {}}},
        {"b5_clean": {"fail_fast_on_leakage": False}},
    )
    assert oracle["oracle_baseline_access_violation_count"] == 0
    assert random["oracle_baseline_access_violation_count"] == 1
