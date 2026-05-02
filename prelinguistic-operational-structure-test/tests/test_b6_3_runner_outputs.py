from pathlib import Path

from src.b6_3_structural_necessity.runner import run_b6_3_structural_necessity, write_b6_3_outputs


def test_b63_runner_outputs_required_splits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"b6_3": {"episodes_per_condition": 1, "conditions": ["clean_reference", "wrong_trace", "delayed_indirect_delay5"]}}
    summary, records, metrics = run_b6_3_structural_necessity(config, seed=0)
    assert {"clean_reference", "wrong_trace", "delayed_indirect_delay5"}.issubset(metrics["conditions"])
    assert records
    assert all(row["sample_count"] > 0 for row in summary if row["policy_name"] == "b63_policy")
    write_b6_3_outputs(summary, records, metrics)
    assert Path("results/b6_3_structural_necessity_summary.csv").exists()
    assert Path("reports/B6_3_STRUCTURAL_NECESSITY_REPORT.md").exists()
    assert Path("reports/B6_3_TRACE_REPAIR_REPORT.md").exists()


def test_b63_full_default_has_no_missing_configured_split():
    summary, _records, metrics = run_b6_3_structural_necessity({"b6_3": {"episodes_per_condition": 1}}, seed=0)
    required = {
        "clean_reference",
        "wrong_trace",
        "wrong_trace_state_ambiguous",
        "missing_trace",
        "ambiguous_trace",
        "low_confidence_trace",
        "missing_mask",
        "hard_hidden_mask",
        "hide_public_state_cue",
        "hide_indirect_target",
        "delayed_indirect_delay5",
        "risk_reward_conflict",
        "spurious_flip",
    }
    assert required.issubset(set(metrics["conditions"]))
    assert all(row["sample_count"] > 0 for row in summary)
