from src.b6_2_hardening.env import make_b62_episode
from src.b6_2_hardening.metrics import mean, score_output


def test_wrong_action_and_inspect_do_not_get_full_score():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 0, "test")
    wrong = {"action": None, "inspect": False}
    scored = score_output(episode, wrong)
    assert scored["risk_constrained_score"] < 1.0


def test_false_safe_commit_score_cap_applies():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 0, "delayed_indirect", delay_steps=1)
    bad = {"action": {"action_type": "apply_local_damping", "region_id": episode["evaluator_ground_truth"]["target_region"]}, "inspect": False}
    scored = score_output(episode, bad)
    assert scored["false_safe_commit"] == 1
    assert scored["risk_constrained_score"] <= 0.20


def test_empty_mean_does_not_default_to_full_credit():
    assert mean([], "risk_constrained_score") == 0.0

