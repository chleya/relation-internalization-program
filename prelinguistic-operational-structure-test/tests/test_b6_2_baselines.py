from src.b6_2_hardening.baselines import always_abstain, oracle, random_policy
from src.b6_2_hardening.env import make_b62_episode
from src.b6_2_hardening.metrics import score_output


def test_always_abstain_has_low_utility_when_action_required():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 2, "test")
    scored = score_output(episode, always_abstain(episode, {}))
    if episode["evaluator_ground_truth"]["expected_action"] is not None:
        assert scored["utility_score"] <= 0.20


def test_random_does_not_exceed_oracle_in_sanity_case():
    episode = make_b62_episode({"b6_2": {"grid_size": 8}}, 3, "test")
    random_score = score_output(episode, random_policy(episode, {}))["risk_constrained_score"]
    oracle_score = score_output(episode, oracle(episode, {}))["risk_constrained_score"]
    assert random_score <= oracle_score

