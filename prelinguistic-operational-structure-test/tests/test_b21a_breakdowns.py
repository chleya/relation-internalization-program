from __future__ import annotations

from src.b21a_degeneracy_audit import (
    compare_region_distributions,
    compute_per_attack_breakdown,
    compute_per_seed_breakdown,
    compute_predicted_region_distribution,
)


def sample_summary():
    return [
        {
            "model": "recurrent_flow_checkpoint_model",
            "seed": "0",
            "false_trace_rejection": "1.0",
            "trace_swap_sensitivity": "1.0",
            "trace_deletion_specificity_ratio": "2.0",
            "multi_source_conflict_resolution": "1.0",
            "noisy_trace_robustness": "1.0",
            "trace_length_extrapolation": "0.9",
            "trace_compression_survival": "0.7",
            "b21_trace_hardening_score": "0.95",
        }
    ]


def test_per_attack_breakdown_has_expected_columns():
    rows = compute_per_attack_breakdown(sample_summary(), [], {})
    assert rows
    for key in ("model", "seed", "attack", "metric_name", "metric_value", "gate", "pass"):
        assert key in rows[0]


def test_per_seed_breakdown_has_expected_columns():
    rows = compute_per_seed_breakdown(sample_summary(), {})
    assert rows
    assert "model" in rows[0]
    assert "b21_trace_hardening_score" in rows[0]


def test_region_distribution_sums_to_one_per_model_attack():
    records = [
        {"model": "a", "attack": "x", "predicted_region": "1"},
        {"model": "a", "attack": "x", "predicted_region": "1"},
        {"model": "a", "attack": "x", "predicted_region": "2"},
    ]
    rows = compute_predicted_region_distribution(records, {})
    total = sum(row["frequency"] for row in rows if row["model"] == "a" and row["attack"] == "x")
    assert abs(total - 1.0) < 1e-9


def test_compare_region_distributions_reports_match_rate():
    records = [
        {"model": "a", "attack": "x", "episode_id": "0", "delay": "4", "intervention_type": "", "predicted_region": "1", "true_trace_region": "1"},
        {"model": "b", "attack": "x", "episode_id": "0", "delay": "4", "intervention_type": "", "predicted_region": "1", "true_trace_region": "1"},
    ]
    dist = compute_predicted_region_distribution(records, {})
    metrics = compare_region_distributions(dist, records)
    assert metrics["cross_model_exact_prediction_match_rate"] == 1.0
    assert metrics["gt_region_match_rate"] == 1.0
