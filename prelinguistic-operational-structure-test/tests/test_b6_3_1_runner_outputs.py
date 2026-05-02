from pathlib import Path

from src.b6_3_1_refinement.runner import CONDITIONS, run_b6_3_1_refinement, write_b6_3_1_outputs
from src.b6_3_1_refinement.result_review import review_b6_3_1_results


def test_b631_default_runner_has_required_conditions():
    summary, records, metrics = run_b6_3_1_refinement({"b6_3_1": {"episodes_per_condition": 1}}, seed=0)
    assert set(CONDITIONS).issubset(set(metrics["conditions"]))
    assert records
    assert all(row["sample_count"] > 0 for row in summary)
    assert metrics["forbidden_reference_count_max"] == 0


def test_b631_writes_outputs_and_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"b6_3_1": {"episodes_per_condition": 1, "conditions": ["wrong_trace_no_public_state", "delay5_credit_buffer_required"]}}
    summary, records, metrics = run_b6_3_1_refinement(config, seed=0)
    write_b6_3_1_outputs(summary, records, metrics)
    review = review_b6_3_1_results()
    assert Path("results/b6_3_1_refinement_summary.csv").exists()
    assert Path("results/b6_3_1_refinement_records.csv").exists()
    assert Path("results/b6_3_1_refinement_metrics.json").exists()
    assert Path("results/b6_3_1_result_review.json").exists()
    assert Path("reports/B6_3_1_WRONG_TRACE_MECHANISM_REFINEMENT.md").exists()
    assert Path("reports/B6_3_1_RESULT_REVIEW.md").exists()
    assert review["integrity"]["forbidden_reference_count_max"] == 0

