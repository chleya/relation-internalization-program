from src.agents_uncertainty import DelayedRelationChainV3Agent
from src.metrics_v32_stress import correlated_failure_recall


def test_correlated_sensor_failure_triggers_takeover():
    assert correlated_failure_recall(DelayedRelationChainV3Agent()) == 1.0

