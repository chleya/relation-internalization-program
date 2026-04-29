from src.agents_uncertainty import DelayedRelationChainV3Agent, OvercautiousTakeoverAgent
from src.metrics_v32_stress import multi_conflict_audit_score


def test_multi_conflict_audit_names_specific_links():
    assert multi_conflict_audit_score(DelayedRelationChainV3Agent()) == 1.0


def test_generic_audit_fails_multi_conflict_specificity():
    assert multi_conflict_audit_score(OvercautiousTakeoverAgent()) < 1.0

