from src.agents_temporal import DelayedRelationChainAgent


def test_delayed_agent_can_edit_multiple_links():
    agent = DelayedRelationChainAgent()
    assert agent.edit_delay("Rainfall -> PorePressure", 2)
    assert agent.edit_delay("PorePressure -> Displacement", 2)
    assert agent.edit_delay("Displacement -> Crack", 2)
    assert agent.edit_delay("Drainage -> PorePressureDown", 2)
    assert agent.edit_delay("Anchoring -> DisplacementDown", 2)
    assert not agent.edit_delay("Unknown -> Link", 2)
