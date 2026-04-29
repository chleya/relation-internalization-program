from src.agents_uncertainty import DelayedRelationChainV3Agent
from src.metrics_v32_stress import long_missing_takeover_recall


def test_long_missing_span_triggers_relation_takeover():
    assert long_missing_takeover_recall(DelayedRelationChainV3Agent()) == 1.0

