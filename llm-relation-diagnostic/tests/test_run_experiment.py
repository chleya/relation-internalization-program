from __future__ import annotations

from pathlib import Path

from src.run_experiment import run


def test_run_experiment_writes_expected_outputs() -> None:
    summary = run("configs/base.yaml")
    by_solver = {row["solver"]: row for row in summary}
    assert by_solver["relation_oracle"]["llm_relation_gated_score"] == 1.0
    for solver in ["global_mapping", "edit_compliance", "missingness_template", "surface_audit"]:
        assert by_solver[solver]["llm_relation_gated_score"] == 0.0
    assert Path("results/llm_relation_records.csv").exists()
    assert Path("results/llm_relation_summary.csv").exists()
    assert Path("reports/LLM_RELATION_DIAGNOSTIC_REPORT.md").exists()


def test_run_experiment_label_writes_separate_outputs() -> None:
    summary = run("configs/base.yaml", solver_override=["relation_oracle"], label="unit_label")
    assert summary[0]["llm_relation_gated_score"] == 1.0
    paths = [
        Path("results/llm_relation_records_unit_label.csv"),
        Path("results/llm_relation_summary_unit_label.csv"),
        Path("results/llm_relation_raw_unit_label.jsonl"),
        Path("reports/LLM_RELATION_DIAGNOSTIC_REPORT_unit_label.md"),
        Path("reports/LLM_RELATION_DIAGNOSTIC_SELF_AUDIT_unit_label.md"),
    ]
    assert all(path.exists() for path in paths)
    for path in paths:
        path.unlink()


def test_run_budgeted_inspect_stress_config() -> None:
    summary = run("configs/budgeted_inspect_stress.yaml")
    by_solver = {row["solver"]: row for row in summary}
    assert by_solver["relation_oracle"]["budgeted_inspect"] == 1.0
    assert by_solver["relation_oracle"]["llm_relation_gated_score"] == 1.0
    assert by_solver["missingness_template"]["llm_relation_gated_score"] == 0.0
    assert Path("results/budgeted_inspect_stress_summary.csv").exists()
    assert Path("reports/BUDGETED_INSPECT_STRESS_REPORT.md").exists()


def test_run_local_edit_behavior_stress_config() -> None:
    summary = run("configs/local_edit_behavior_stress.yaml")
    by_solver = {row["solver"]: row for row in summary}
    assert by_solver["relation_oracle"]["local_edit_locality"] == 1.0
    assert by_solver["relation_oracle"]["llm_relation_gated_score"] == 1.0
    assert by_solver["edit_compliance"]["llm_relation_gated_score"] == 0.0
    assert by_solver["edit_no_behavior"]["llm_relation_gated_score"] == 0.0
    assert Path("results/local_edit_behavior_stress_summary.csv").exists()
    assert Path("reports/LOCAL_EDIT_BEHAVIOR_STRESS_REPORT.md").exists()


def test_run_audit_correctness_stress_config() -> None:
    summary = run("configs/audit_correctness_stress.yaml")
    by_solver = {row["solver"]: row for row in summary}
    assert by_solver["relation_oracle"]["audit_correctness"] == 1.0
    assert by_solver["relation_oracle"]["llm_relation_gated_score"] == 1.0
    for solver in ["surface_audit", "missingness_template", "reverse_audit", "outcome_audit"]:
        assert by_solver[solver]["llm_relation_gated_score"] == 0.0
    assert Path("results/audit_correctness_stress_summary.csv").exists()
    assert Path("reports/AUDIT_CORRECTNESS_STRESS_REPORT.md").exists()
