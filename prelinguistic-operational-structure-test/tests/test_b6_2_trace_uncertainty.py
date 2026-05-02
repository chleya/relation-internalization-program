from src.b6_2_hardening.env import make_b62_episode
from src.b6_2_hardening.metrics import score_output
from src.b6_2_hardening.policy import b62_policy


def test_wrong_trace_does_not_silently_keep_oracle_direct_action():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 1, "wrong_trace", trace_mode="wrong")
    output = b62_policy(episode, {})
    assert output["trace_conflict"]["trace_conflict_detected"] is True
    assert output["trace_repair"]["trace_confidence_downgraded"] is True
    assert output["inspect"] is True
    scored = score_output(episode, output)
    assert scored["risk_constrained_score"] <= 1.0


def test_missing_trace_uses_fallback_and_records_uncertainty():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 2, "wrong_trace", trace_mode="missing")
    output = b62_policy(episode, {})
    assert output["trace_conflict"]["trace_conflict_detected"] is True
    assert output["inspect"] is True
    assert output["target_source"] in {"state_fallback", "trace_repair"}


def test_poisoned_evaluator_does_not_change_trace_repair():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 5, "wrong_trace", trace_mode="wrong")
    output = b62_policy(episode, {})
    episode["evaluator_ground_truth"]["expected_action"] = None
    episode["evaluator_ground_truth"]["condition"] = "poisoned"
    poisoned = b62_policy(episode, {})
    assert output["action"] == poisoned["action"]
    assert output["trace_repair"] == poisoned["trace_repair"]


def test_removed_public_state_cue_degrades_repair_candidate():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 6, "wrong_trace", trace_mode="wrong")
    output = b62_policy(episode, {})
    assert output["trace_repair"]["trace_repaired"] is True
    visible = episode["model_input"]["visible_state"]
    visible["state_target_confidence"] = 0.0
    visible["state_target_candidates"] = []
    visible["feedback_history_region"] = None
    visible["feedback_confidence"] = 0.0
    degraded = b62_policy(episode, {})
    assert degraded["trace_repair"]["trace_repaired"] is False
