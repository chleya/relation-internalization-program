from src.b6_hardening.hidden_risk_env import make_hardening_episode
from src.b6_hardening.spurious_safe_cue import causal_safety_feature_reliance, spurious_cue_dependency


def test_spurious_safe_cue_flips_in_hard_mode():
    clean = make_hardening_episode("spurious_safe_cue", 0, {"b6_1": {"grid_size": 8}}, spurious_mode="clean_correlated")
    flipped = make_hardening_episode("spurious_safe_cue", 0, {"b6_1": {"grid_size": 8}}, spurious_mode="hard_flipped")
    assert clean["model_input"]["visible_state"]["spurious_safe_feature"] != flipped["model_input"]["visible_state"]["spurious_safe_feature"]


def test_spurious_dependency_metric_valid_range():
    records = [
        {"spurious_safe_feature": 1, "abstained": 0, "unsafe_action": 0},
        {"spurious_safe_feature": 0, "abstained": 1, "unsafe_action": 0},
    ]
    assert 0.0 <= spurious_cue_dependency(records) <= 1.0
    assert causal_safety_feature_reliance(records) == 1.0

