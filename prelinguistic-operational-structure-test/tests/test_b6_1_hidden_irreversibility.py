from src.b6_hardening.hardening_baselines import mask_only_policy
from src.b6_hardening.hardening_metrics import score_policy_output
from src.b6_hardening.hardening_policy import hardening_policy
from src.b6_hardening.hidden_risk_env import make_hidden_irreversibility_episode


def test_hidden_irreversibility_hides_public_flag_but_keeps_latent_marker():
    episode = make_hidden_irreversibility_episode(4, {"b6_1": {"grid_size": 8}}, hidden=True)
    target = episode["evaluator_ground_truth"]["target_region"]
    assert episode["model_input"]["actionability_mask"][target]["irreversible"] is False
    assert episode["model_input"]["visible_state"]["latent_risk_marker"] == "irreversible"


def test_hardening_policy_rejects_hidden_irreversible_action():
    episode = make_hidden_irreversibility_episode(4, {"b6_1": {"grid_size": 8}}, hidden=True)
    output = hardening_policy(episode, {})
    scored = score_policy_output(episode, output)
    assert output["abstained"] is True
    assert scored["irreversible_action"] == 0


def test_mask_only_policy_fails_hidden_irreversibility_sanity_case():
    episode = make_hidden_irreversibility_episode(4, {"b6_1": {"grid_size": 8}}, hidden=True)
    output = mask_only_policy(episode, {})
    scored = score_policy_output(episode, output)
    assert scored["irreversible_action"] == 1

