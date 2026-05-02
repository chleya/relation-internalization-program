from src.b6_3_structural_necessity.metrics import structural_score, summarize_policy


def test_structural_score_zeroes_on_forbidden_reference():
    row = {
        "forbidden_reference_count": 1,
        "invalid_metric_count": 0,
        "trace_conflict_detection_accuracy": 1.0,
        "trace_confidence_downgrade_rate": 1.0,
        "gain_over_state_only": 1.0,
        "gain_over_mask_only": 1.0,
    }
    assert structural_score(row, 0.5) == 0.0


def test_empty_summary_is_invalid_not_full_score():
    row = summarize_policy(
        [],
        {"condition": "empty", "seed": 0, "policy_name": "b63_policy"},
        {},
        1.0,
        "reference",
        {"forbidden_reference_count": 0, "poisoned_ground_truth_invariance_pass": 1, "policy_uses_model_input_only": True},
    )
    assert row["sample_count"] == 0
    assert row["invalid_metric_count"] == 1
    assert row["structural_necessity_score"] == 0.0
