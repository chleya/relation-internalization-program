from src.agents_temporal import DelayedRelationChainAgent, LearnedDelayedLinksAgent, SurfaceTemporalAgent
from src.env_temporal import generate_sequence
from src.metrics_temporal import delay_edit_success, temporal_audit_score, train_agent


def test_delay_edit_success_for_oracle():
    assert delay_edit_success(DelayedRelationChainAgent()) == 1.0


def test_temporal_audit_rejects_surface_agent():
    assert temporal_audit_score(SurfaceTemporalAgent()) == 0.0
    assert temporal_audit_score(DelayedRelationChainAgent()) == 1.0


def test_learned_delayed_links_learns_delay_one():
    agent = LearnedDelayedLinksAgent()
    for seed in range(30):
        agent.observe_sequence(generate_sequence(seed, mode="base"))
    assert agent.delay_map["Rainfall -> PorePressure"] == 1
