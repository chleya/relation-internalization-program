from pathlib import Path

from src.b6_4_2_combined_remap_refinement.combined_remap_env import CONDITIONS
from src.b6_4_2_combined_remap_refinement.result_review import review_b6_4_2_results
from src.b6_4_2_combined_remap_refinement.runner import run_b6_4_2_combined_refinement, write_b6_4_2_outputs


def test_b642_runner_has_all_conditions():
    summary, records, metrics = run_b6_4_2_combined_refinement({"b6_4_2": {"episodes_per_condition": 1}}, seed=0)
    assert set(CONDITIONS).issubset(set(metrics["conditions"]))
    assert records
    assert all(row["sample_count"] > 0 for row in summary)
    assert metrics["forbidden_reference_count_max"] == 0


def test_b642_outputs_and_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary, records, metrics = run_b6_4_2_combined_refinement({"b6_4_2": {"episodes_per_condition": 1}}, seed=0)
    write_b6_4_2_outputs(summary, records, metrics)
    review = review_b6_4_2_results()
    assert Path("results/b6_4_2_combined_refinement_summary.csv").exists()
    assert Path("results/b6_4_2_combined_refinement_records.csv").exists()
    assert Path("results/b6_4_2_combined_refinement_metrics.json").exists()
    assert Path("results/b6_4_2_result_review.json").exists()
    assert Path("reports/B6_4_2_COMBINED_REMAP_REFINEMENT_REPORT.md").exists()
    assert Path("reports/B6_4_2_RESULT_REVIEW.md").exists()
    assert review["integrity"]["forbidden_reference_count_max"] == 0
