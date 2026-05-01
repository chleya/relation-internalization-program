from src.b51_value_leakage_audit import audit_b5_value_leakage


def test_value_leakage_detects_forbidden_keys():
    result = audit_b5_value_leakage({"ground_truth": {"oracle_closed_loop_plan": {}}}, {}, {})
    assert result["value_leakage_count"] > 0


def test_value_leakage_clean_input_zero():
    result = audit_b5_value_leakage({"trace_state": {"region": 1}}, {"action": "x"}, {})
    assert result["value_leakage_count"] == 0
