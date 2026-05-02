from src.b6_hardening.hidden_risk_env import make_hardening_episode
from src.b6_hardening.noisy_mask import apply_mask_noise, inject_actionability_noise, noisy_mask_metrics


def test_noisy_mask_can_mark_unsafe_as_safe():
    info = {"unsafe": True, "risk_cost": 1.0, "blocked_actions": []}
    apply_mask_noise(info, "unsafe_as_safe")
    assert info["unsafe"] is False
    assert info["risk_cost"] == 0.0


def test_inject_actionability_noise_is_deterministic():
    mask = {0: {"unsafe": True, "risk_cost": 1.0, "blocked_actions": [], "intervenable": True, "directly_intervenable": True, "irreversible": False, "inspectable": True}}
    first = inject_actionability_noise(mask, 1.0, seed=0)
    second = inject_actionability_noise(mask, 1.0, seed=0)
    assert first == second


def test_noisy_mask_metrics_returns_required_fields():
    records = [
        {"risk_constrained_score": 1.0, "unsafe_action": 0, "abstained": 0, "mask_noise_rate": 0.0},
        {"risk_constrained_score": 0.5, "unsafe_action": 1, "abstained": 1, "mask_noise_rate": 0.4},
    ]
    metrics = noisy_mask_metrics(records)
    assert 0.0 <= metrics["noisy_mask_score"] <= 1.0
    assert "mask_noise_degradation_slope" in metrics


def test_noisy_mask_episode_keeps_evaluator_separate():
    episode = make_hardening_episode("noisy_mask", 3, {"b6_1": {"grid_size": 8}}, mask_noise_rate=0.4)
    assert "evaluator_ground_truth" not in episode["model_input"]
    assert episode["evaluator_ground_truth"]["condition"] == "noisy_mask"

