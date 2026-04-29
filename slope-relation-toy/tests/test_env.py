import numpy as np

from src.env import evaluate_chain, reward_for
from src.evaluate import corrupt_monitoring_observation


def test_chain_high_rain_poor_drainage_needs_drain():
    state = evaluate_chain(
        {
            "rainfall": "high",
            "drainage": "poor",
            "anchoring": "none",
            "toe_excavation": "no",
            "monitoring": "dense",
        }
    )
    assert state.pore_pressure == "high"
    assert state.risk == "medium"
    assert state.optimal_action == "drain"


def test_reward_penalizes_monitoring_high_risk():
    assert reward_for("monitor", "stop_work", "high") == -1.0


def test_corrupt_monitoring_observation_can_flip_monitoring():
    context = {
        "rainfall": "low",
        "drainage": "good",
        "anchoring": "none",
        "toe_excavation": "no",
        "monitoring": "dense",
    }
    observed = corrupt_monitoring_observation(context, np.random.default_rng(0), noise_rate=1.0)
    assert observed["monitoring"] == "sparse"
    assert context["monitoring"] == "dense"
