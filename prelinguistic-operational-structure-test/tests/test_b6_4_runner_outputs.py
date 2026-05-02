from pathlib import Path

from src.b6_4_transfer_generalization.remap_configs import CONDITIONS
from src.b6_4_transfer_generalization.result_review import review_b6_4_results
from src.b6_4_transfer_generalization.runner import run_b6_4_transfer, write_b6_4_outputs


def test_b64_runner_includes_all_remaps():
    summary, records, metrics = run_b6_4_transfer({"b6_4": {"episodes_per_condition": 1}}, seed=0)
    assert set(CONDITIONS).issubset(set(metrics["conditions"]))
    assert records
    assert all(row["sample_count"] > 0 for row in summary)
    assert metrics["forbidden_reference_count_max"] == 0


def test_b64_writes_outputs_and_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"b6_4": {"episodes_per_condition": 1, "conditions": ["clean_reference", "visual_remap", "combined_remap"]}}
    summary, records, metrics = run_b6_4_transfer(config, seed=0)
    write_b6_4_outputs(summary, records, metrics)
    review = review_b6_4_results()
    assert Path("results/b6_4_transfer_summary.csv").exists()
    assert Path("results/b6_4_transfer_records.csv").exists()
    assert Path("results/b6_4_transfer_metrics.json").exists()
    assert Path("results/b6_4_result_review.json").exists()
    assert Path("reports/B6_4_TRANSFER_GENERALIZATION_REPORT.md").exists()
    assert Path("reports/B6_4_RESULT_REVIEW.md").exists()
    assert review["integrity"]["forbidden_reference_count_max"] == 0

