from __future__ import annotations

from src.agents import RelationAgent
from src.metrics import counterfactual_accuracy, edit_success, relation_recovery, train_agent


def test_relation_agent_learns_relations() -> None:
    agent = RelationAgent()
    train_agent(agent, seed=0, steps=250)
    assert relation_recovery(agent) >= 0.7


def test_relation_agent_counterfactual_and_edit() -> None:
    agent = RelationAgent()
    train_agent(agent, seed=1, steps=250)
    assert counterfactual_accuracy(agent) >= 0.8
    assert edit_success(agent) == 1.0

