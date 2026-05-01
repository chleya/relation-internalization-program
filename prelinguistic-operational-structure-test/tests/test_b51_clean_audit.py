import yaml

from src.b51_clean_audit_runner import b51_clean_closed_loop_audit_score, run_b51_clean_closed_loop_audit


def test_run_b51_clean_audit_keeps_leakage_zero(tmp_path):
    clean_config = yaml.safe_load(open("configs/b5_clean_closed_loop.yaml", encoding="utf-8"))
    clean_config["b5_clean"]["max_eval_episodes"] = 4
    clean_path = tmp_path / "b5_clean_closed_loop.yaml"
    clean_path.write_text(yaml.safe_dump(clean_config), encoding="utf-8")
    config = yaml.safe_load(open("configs/b51_clean_closed_loop_audit.yaml", encoding="utf-8"))
    config["clean_b5_config"] = str(clean_path)
    summary, records, leakage = run_b51_clean_closed_loop_audit(config, seed=0)
    assert summary
    assert records
    assert sum(int(row["leakage_count"]) for row in leakage) == 0
    assert all(float(row["value_leakage_count"]) == 0.0 for row in summary)


def test_b51_clean_score_zero_if_leakage_injected():
    metrics = {
        "value_leakage_count": 1.0,
        "oracle_plan_usage_rate": 0.0,
        "oracle_trace_update_usage_rate": 0.0,
        "oracle_feedback_revision_usage_rate": 0.0,
        "cross_model_exact_plan_match_rate": 0.0,
        "exact_all_model_same_plan_rate": 0.0,
        "model_gain_over_scripted_update": 0.2,
        "feedback_revision_over_scripted_ratio": 2.0,
    }
    assert b51_clean_closed_loop_audit_score(metrics, {"value_leakage_count": 0.0}) == 0.0
