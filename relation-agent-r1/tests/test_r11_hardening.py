from __future__ import annotations

from src.agents import RelationAgent
from src.env import REVERSAL_LINKS, TRUE_LINKS, ProcessWorld
from src.metrics_r11 import (
    FALSE_CANDIDATE_LINKS,
    active_discovery_score,
    add_confounders,
    candidate_expansion_precision,
    make_r11_agent,
    reversal_adaptation,
    train_agent_mode,
)
from src.run_r11_hardening import run


def test_rule_reversal_changes_rainfall_relation() -> None:
    env = ProcessWorld(seed=0, mode="rule_reversal")
    high = env._evaluate_process(
        {
            "rainfall": "high",
            "load": "high",
            "drainage": "poor",
            "support": "absent",
            "pore_pressure": "normal",
            "displacement": "normal",
            "risk": "low",
            "warning": "normal",
        }
    )
    low = env._evaluate_process({**high, "rainfall": "low"})
    assert high["pore_pressure"] == "normal"
    assert low["pore_pressure"] == "high"


def test_relation_agent_ignores_marked_nuisance_candidates() -> None:
    agent = RelationAgent(
        candidate_links=TRUE_LINKS | REVERSAL_LINKS | FALSE_CANDIDATE_LINKS,
        ignored_features={"season", "contractor", "warning"},
    )
    train_agent_mode(agent, seed=0, steps=250, confounder_correlation=0.9)
    assert not (agent.learned_links() & FALSE_CANDIDATE_LINKS)


def test_no_explore_fails_active_discovery() -> None:
    config = {
        "train_steps": 60,
        "relation_agent": {
            "min_support": 4,
            "min_confidence": 0.65,
            "decay": 0.98,
            "ignored_features": ["warning", "season", "contractor"],
            "physical_action_cost": 0.25,
            "inspect_cost": 0.05,
        },
    }
    assert active_discovery_score("relation_no_explore", 0, config) == 0.0


def test_r11_run_only_relation_agent_passes_gate() -> None:
    rows = run("configs/r11_hardening.yaml")
    by_agent = {row["agent"]: row for row in rows}
    assert by_agent["relation_agent"]["hardening_r11_gated_score"] > 0.7
    for agent in ["random", "shortcut", "passive_memory", "relation_no_explore"]:
        assert by_agent[agent]["hardening_r11_gated_score"] == 0.0
