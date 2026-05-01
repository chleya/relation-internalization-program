from src.b6_hardening.hidden_risk_env import make_hardening_episode
from src.b6_hardening.risk_reward_conflict import choose_risk_reward_action, risk_reward_tradeoff_score


def test_risk_reward_policy_rejects_high_risk_reward():
    action = choose_risk_reward_action(0.9, {"unsafe_high": 1.0, "medium_reversible": 0.75})
    assert action["action_type"] == "abstain"


def test_risk_reward_episode_has_reward_estimate():
    episode = make_hardening_episode("risk_reward_conflict", 0, {"b6_1": {"grid_size": 8}})
    assert "reward_estimate" in episode["model_input"]["visible_state"]


def test_risk_reward_tradeoff_score_valid_range():
    value = risk_reward_tradeoff_score([
        {"unsafe_action": 0, "irreversible_action": 0, "utility_score": 0.8},
        {"unsafe_action": 1, "irreversible_action": 0, "utility_score": 1.0},
    ])
    assert 0.0 <= value <= 1.0

