from src.agents_uncertainty import DelayedRelationChainV3Agent, OvercautiousTakeoverAgent
from src.metrics_v31_hardening import audit_specificity, conflicting_evidence_takeover


def test_relation_agent_takes_over_for_conflicting_evidence():
    assert conflicting_evidence_takeover(DelayedRelationChainV3Agent()) == 1.0


def test_relation_agent_audit_is_specific():
    assert audit_specificity(DelayedRelationChainV3Agent()) == 1.0


def test_generic_takeover_audit_fails_specificity():
    assert audit_specificity(OvercautiousTakeoverAgent()) < 1.0

