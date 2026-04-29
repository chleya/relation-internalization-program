from src.agents_uncertainty import DelayedRelationChainV3Agent, OvercautiousTakeoverAgent
from src.metrics_v31_hardening import irrelevant_missing_rejection


def test_irrelevant_missing_sensor_does_not_force_takeover_for_relation_agent():
    assert irrelevant_missing_rejection(DelayedRelationChainV3Agent()) == 1.0


def test_overcautious_agent_fails_irrelevant_missing_rejection():
    assert irrelevant_missing_rejection(OvercautiousTakeoverAgent()) == 0.0

