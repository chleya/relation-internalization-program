from pathlib import Path

from src.b6_2_hardening.runner import run_b6_2_hardening, write_b6_2_outputs


def test_runner_generates_explicit_condition_splits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"b6_2": {"grid_size": 8, "episodes_per_condition": 2, "conditions": ["test", "missing_mask", "delayed_indirect"]}}
    summary, records, metrics = run_b6_2_hardening(config, seed=0)
    conditions = {row["condition"] for row in summary}
    assert {"test", "missing_mask", "delayed_indirect"}.issubset(conditions)
    assert metrics["conditions"] == ["delayed_indirect", "missing_mask", "test"]
    write_b6_2_outputs(summary, records, metrics)
    assert Path("results/b6_2_hardening_summary.csv").exists()
    assert Path("results/b6_2_audit_summary.json").exists()
    assert Path("reports/B6_2_FALLBACK_RISK_AND_DELAYED_CREDIT.md").exists()


def test_default_runner_generates_second_pass_focus_splits():
    summary, _records, _metrics = run_b6_2_hardening({"b6_2": {"grid_size": 8, "episodes_per_condition": 1}}, seed=0)
    conditions = {row["condition"] for row in summary}
    required = {
        "missing_trace",
        "ambiguous_trace",
        "low_confidence_trace",
        "hard_hidden_mask",
        "risk_reward_conflict",
        "wrong_trace_state_ambiguous",
    }
    assert required.issubset(conditions)
    b62_rows = [row for row in summary if row["policy_name"] == "b62_policy" and row["condition"] in required]
    assert b62_rows
    assert all(row["sample_count"] > 0 for row in b62_rows)


def test_requested_split_missing_would_be_observable():
    config = {"b6_2": {"grid_size": 8, "episodes_per_condition": 1, "conditions": ["missing_mask"]}}
    summary, _records, _metrics = run_b6_2_hardening(config, seed=0)
    assert {row["condition"] for row in summary} == {"missing_mask"}


def test_empty_condition_is_marked_invalid_not_full_score():
    config = {"b6_2": {"grid_size": 8, "episodes_per_condition": 0, "conditions": ["missing_mask"]}}
    summary, _records, _metrics = run_b6_2_hardening(config, seed=0)
    row = summary[0]
    assert row["sample_count"] == 0
    assert row["invalid_metric_count"] == 1
    assert row["b62_score"] == 0.0
