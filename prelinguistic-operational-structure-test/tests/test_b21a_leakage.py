from __future__ import annotations

from src.b21a_degeneracy_audit import audit_batch_for_ground_truth_leakage, audit_structure_for_ground_truth_leakage, recursive_forbidden_key_find


def test_forbidden_keys_detected_recursively():
    obj = {"safe": {"ground_truth": {"true_trace_region": 12}}}
    findings = recursive_forbidden_key_find(obj, ["ground_truth", "true_trace_region"])
    keys = {finding["forbidden_key"] for finding in findings}
    assert "ground_truth" in keys
    assert "true_trace_region" in keys


def test_safe_batch_passes_leakage_audit():
    batch = {"past_frames": "array", "future_horizon": 12, "frame_size": 64, "grid_size": 8}
    result = audit_batch_for_ground_truth_leakage(batch)
    assert result["leakage_count"] == 0


def test_structure_leakage_count_is_correct():
    structure = {"applicable": True, "critical_inspection_region": 4}
    result = audit_structure_for_ground_truth_leakage(structure)
    assert result["leakage_count"] == 1
