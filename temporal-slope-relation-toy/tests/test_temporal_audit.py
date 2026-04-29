from src.agents_temporal import DelayedRelationChainAgent, InstantRelationChainAgent


def test_temporal_audit_contains_time_indexes():
    audit = DelayedRelationChainAgent().audit_temporal_relations()
    joined = "\n".join(audit)
    assert "[t-" in joined
    assert "Rainfall" in joined
    assert "PorePressure" in joined
    assert "Displacement" in joined
    assert "Drainage" in joined
    assert "Anchoring" in joined


def test_instant_audit_does_not_pass_time_index_requirement():
    assert "[t-" not in "\n".join(InstantRelationChainAgent().audit_temporal_relations())
