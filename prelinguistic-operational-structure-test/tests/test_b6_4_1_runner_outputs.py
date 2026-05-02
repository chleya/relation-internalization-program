from pathlib import Path

from src.b6_4_1_transfer_hardening.hard_remap_env import CONDITIONS
from src.b6_4_1_transfer_hardening.result_review import review_b6_4_1_results
from src.b6_4_1_transfer_hardening.runner import run_b6_4_1_transfer_hardening, write_b6_4_1_outputs


def test_b641_runner_has_all_conditions():
    summary, records, metrics = run_b6_4_1_transfer_hardening({"b6_4_1": {"episodes_per_condition": 1}}, seed=0)
    assert set(CONDITIONS).issubset(set(metrics["conditions"]))
    assert records
    assert all(row["sample_count"] > 0 for row in summary)
    assert metrics["forbidden_reference_count_max"] == 0


def test_b641_outputs_and_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary, records, metrics = run_b6_4_1_transfer_hardening({"b6_4_1": {"episodes_per_condition": 1}}, seed=0)
    write_b6_4_1_outputs(summary, records, metrics)
    review = review_b6_4_1_results()
    assert Path("results/b6_4_1_transfer_hardening_summary.csv").exists()
    assert Path("results/b6_4_1_transfer_hardening_records.csv").exists()
    assert Path("results/b6_4_1_transfer_hardening_metrics.json").exists()
    assert Path("results/b6_4_1_result_review.json").exists()
    assert Path("reports/B6_4_1_TRANSFER_HARDENING_REPORT.md").exists()
    assert Path("reports/B6_4_1_RESULT_REVIEW.md").exists()
    assert review["integrity"]["forbidden_reference_count_max"] == 0

