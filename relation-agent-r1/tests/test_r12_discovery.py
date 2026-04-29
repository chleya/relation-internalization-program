from __future__ import annotations

from src.agents import DiscoveryRelationAgent
from src.env import ProcessWorld
from src.metrics_r12 import NEW_SUPPORT_PRESSURE_LINKS, train_in_mode, train_with_persistent_nuisance
from src.run_r12_discovery import run


def test_new_support_pressure_mode_adds_new_link_behavior() -> None:
    env = ProcessWorld(seed=0, mode="new_support_pressure_link")
    state = {
        "rainfall": "high",
        "load": "low",
        "drainage": "poor",
        "support": "present",
        "pore_pressure": "high",
        "displacement": "normal",
        "risk": "low",
        "warning": "normal",
    }
    evaluated = env._evaluate_process(state)
    assert evaluated["pore_pressure"] == "normal"


def test_discovery_agent_learns_link_not_in_true_links() -> None:
    agent = DiscoveryRelationAgent()
    train_in_mode(agent, seed=0, steps=350, mode="new_support_pressure_link")
    assert agent.learned_links() & NEW_SUPPORT_PRESSURE_LINKS


def test_discovery_agent_does_not_emit_unmarked_nuisance_links() -> None:
    agent = DiscoveryRelationAgent()
    train_with_persistent_nuisance(agent, seed=1, steps=300, correlation=0.9)
    learned = agent.learned_links()
    assert not {link for link in learned if "season=" in link[0] or "contractor=" in link[0]}


def test_r12_run_only_discovery_agent_passes_gate() -> None:
    rows = run("configs/r12_discovery.yaml")
    by_agent = {row["agent"]: row for row in rows}
    assert by_agent["discovery_relation_agent"]["discovery_r12_gated_score"] > 0.7
    for agent in ["random", "shortcut", "passive_memory", "relation_agent"]:
        assert by_agent[agent]["discovery_r12_gated_score"] == 0.0
