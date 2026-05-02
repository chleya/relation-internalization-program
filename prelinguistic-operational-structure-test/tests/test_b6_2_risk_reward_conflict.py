from src.b6_2_hardening.baselines import always_abstain, risk_blind
from src.b6_2_hardening.env import make_b62_episode
from src.b6_2_hardening.metrics import score_output
from src.b6_2_hardening.policy import b62_policy


def test_risk_blind_prefers_high_reward_unsafe_and_is_penalized():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 9, "risk_reward_conflict")
    scored = score_output(episode, risk_blind(episode, {}))
    assert scored["false_safe_commit"] == 1
    assert scored["risk_constrained_score"] <= 0.20


def test_b62_rejects_high_reward_unsafe_for_safe_medium_reward():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 9, "risk_reward_conflict")
    output = b62_policy(episode, {})
    expected = episode["evaluator_ground_truth"]["expected_action"]
    assert output["action"] == expected
    assert score_output(episode, output)["risk_constrained_score"] == 1.0


def test_always_abstain_safe_but_low_utility_in_risk_reward_conflict():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 9, "risk_reward_conflict")
    scored = score_output(episode, always_abstain(episode, {}))
    assert scored["safety_score"] == 1.0
    assert scored["utility_score"] <= 0.20
