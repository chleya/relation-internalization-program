from __future__ import annotations

from src.agents import UncertaintyDiscoveryAgent
from src.metrics_r12 import train_in_mode
from src.metrics_r2 import needs_inspection
from src.run_r2_partial_observability import run


def test_uncertainty_agent_inspects_missing_critical_node() -> None:
    agent = UncertaintyDiscoveryAgent()
    train_in_mode(agent, seed=0, steps=300, mode="base")
    state = {
        "rainfall": "high",
        "load": "high",
        "drainage": "poor",
        "support": "absent",
        "pore_pressure": "unknown",
        "displacement": "high",
        "risk": "high",
        "warning": "warning",
    }
    assert agent.act(state) == "inspect"


def test_uncertainty_audit_names_relation_link() -> None:
    agent = UncertaintyDiscoveryAgent()
    state = {
        "rainfall": "high",
        "load": "high",
        "drainage": "poor",
        "support": "absent",
        "pore_pressure": "normal",
        "displacement": "high",
        "risk": "high",
        "warning": "warning",
    }
    audit = agent.uncertainty_audit(state)
    assert audit
    assert "link" in audit[0]
    assert audit[0]["recommended_action"] == "inspect"


def test_needs_inspection_on_conflict() -> None:
    assert needs_inspection({"displacement": "normal", "risk": "high"})
    assert needs_inspection({"rainfall": "high", "drainage": "poor", "pore_pressure": "normal"})


def test_r2_run_only_uncertainty_agent_passes_gate() -> None:
    rows = run("configs/r2_partial_observability.yaml")
    by_agent = {row["agent"]: row for row in rows}
    assert by_agent["uncertainty_discovery_agent"]["partial_r2_gated_score"] > 0.8
    for agent in ["random", "shortcut", "passive_memory", "discovery_relation_agent"]:
        assert by_agent[agent]["partial_r2_gated_score"] == 0.0
