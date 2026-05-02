from src.b6_4_2_combined_remap_refinement.combined_metrics import score_output
from src.b6_4_2_combined_remap_refinement.combined_remap_env import make_b642_episode
from src.b6_4_2_combined_remap_refinement.runner import run_b6_4_2_combined_refinement


def test_b642_wrong_decision_does_not_get_full_score():
    episode = make_b642_episode({"b6_4_2": {}}, 0, "pair_visual_risk")
    out = {
        "action": {"action_type": "apply_local_damping", "region_id": episode["evaluator_ground_truth"]["target_region"] + 1},
        "delayed_credit": {"credit_assigned": False},
        "hidden_indirect": {"success": False},
    }
    scored = score_output(episode, out)
    assert scored["combined_refinement_score"] < 1.0


def test_b642_empty_condition_does_not_silently_pass():
    summary, _, metrics = run_b6_4_2_combined_refinement({"b6_4_2": {"episodes_per_condition": 0}}, seed=0)
    assert any(row["no_sample_metric_count"] == 1 for row in summary)
    assert metrics["no_sample_metric_count_total"] > 0


def test_b642_combined_gate_stays_in_b6_refinement_when_gap_remains():
    _, _, metrics = run_b6_4_2_combined_refinement({"b6_4_2": {"episodes_per_condition": 4}}, seed=0)
    assert metrics["combined_oracle_gap"] > 0.15
    assert metrics["gate_judgment"] == "STAY_IN_B6_REFINEMENT"
