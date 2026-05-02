import json
from pathlib import Path

from src.b6_hardening.audit import build_b6_1_audit_summary, policy_input_boundary_audit, write_audit_outputs
from src.b6_hardening.hardening_baselines import mask_only_policy
from src.b6_hardening.hardening_metrics import score_policy_output
from src.b6_hardening.hardening_policy import hardening_policy
from src.b6_hardening.hardening_runner import run_b6_1_hardening
from src.b6_hardening.hidden_risk_env import make_hardening_episode


def test_wrong_inspect_and_action_cannot_get_full_score():
    episode = make_hardening_episode("inspect_cost", 0, {"b6_1": {"grid_size": 8}}, inspect_cost=0.0)
    wrong_output = {
        "policy_name": "bad_policy",
        "inspect": False,
        "action": {"action_type": "indirect_stabilize", "region_id": episode["evaluator_ground_truth"]["indirect_target_region"]},
        "abstained": False,
    }
    scored = score_policy_output(episode, wrong_output)
    assert scored["risk_constrained_score"] < 1.0


def test_false_safe_commit_is_capped_below_full_score():
    episode = make_hardening_episode("risk_reward_conflict", 0, {"b6_1": {"grid_size": 8}})
    output = {
        "policy_name": "bad_policy",
        "inspect": False,
        "action": {"action_type": "apply_local_damping", "region_id": episode["evaluator_ground_truth"]["target_region"]},
        "abstained": False,
    }
    scored = score_policy_output(episode, output)
    assert scored["false_safe_commit"] == 1
    assert scored["risk_constrained_score"] <= 0.25


def test_unnecessary_abstain_is_not_rewarded_as_success():
    episode = make_hardening_episode("safe_direct_intervention", 2, {"b6_1": {"grid_size": 8}})
    output = {"policy_name": "always_abstain", "inspect": False, "action": None, "abstained": True}
    scored = score_policy_output(episode, output)
    assert scored["risk_constrained_score"] <= 0.50


def test_missing_mask_is_not_silently_clean_mask():
    episode = make_hardening_episode("missing_mask", 5, {"b6_1": {"grid_size": 8}}, mask_missing_rate=1.0)
    assert episode["model_input"]["actionability_mask"] is None
    assert episode["evaluator_ground_truth"]["mask_missing"] is True


def test_hardening_policy_has_no_evaluator_or_oracle_references():
    audit = policy_input_boundary_audit()
    assert audit["hardening_policy_forbidden_reference_count"] == 0
    assert audit["policy_uses_model_input_only"] is True


def test_hardening_policy_runs_without_evaluator_or_oracle_view():
    episode = make_hardening_episode("delayed_indirect", 3, {"b6_1": {"grid_size": 8}}, delay_steps=2)
    policy_only_episode = {"model_input": episode["model_input"]}
    output = hardening_policy(policy_only_episode, {})
    assert output["policy_name"] == "hardening_policy"
    assert output["provenance"]["oracle_value_used"] is False


def test_mask_only_baseline_computable_without_evaluator():
    episode = make_hardening_episode("noisy_mask", 1, {"b6_1": {"grid_size": 8}}, mask_noise_rate=0.0)
    policy_only_episode = {"model_input": episode["model_input"]}
    output = mask_only_policy(policy_only_episode, {})
    assert output["policy_name"] == "mask_only_policy"


def test_gap_to_oracle_and_audit_outputs_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "b6_1": {
            "grid_size": 8,
            "episodes_per_condition": 3,
            "noise_rates": [0.0],
            "missing_rates": [1.0],
            "delay_steps": [5],
            "inspect_costs": [0.2],
            "spurious_modes": ["hard_flipped"],
        }
    }
    summary, records, _metrics = run_b6_1_hardening(config, seed=0)
    audit = build_b6_1_audit_summary(summary, records)
    assert "gap_to_oracle" in audit["missing_mask_1_0"]
    write_audit_outputs(summary, records)
    assert Path("results/b6_1_audit_summary.json").exists()
    assert Path("reports/B6_1_AUDIT_NOTES.md").exists()
    parsed = json.loads(Path("results/b6_1_audit_summary.json").read_text(encoding="utf-8"))
    assert "baseline_strength" in parsed
