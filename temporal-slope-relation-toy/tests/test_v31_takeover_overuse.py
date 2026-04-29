from src.agents_uncertainty import DelayedRelationChainV3Agent, OvercautiousTakeoverAgent
from src.metrics_v31_hardening import takeover_overuse_control


def test_relation_agent_rejects_takeover_overuse():
    assert takeover_overuse_control(DelayedRelationChainV3Agent()) == 1.0


def test_overcautious_agent_fails_takeover_overuse_control():
    assert takeover_overuse_control(OvercautiousTakeoverAgent()) == 0.0

