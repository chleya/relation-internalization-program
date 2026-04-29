from __future__ import annotations

from src.run_r1_experiment import run


def test_r1_run_only_relation_agent_passes_gate() -> None:
    rows = run("configs/r1.yaml")
    by_agent = {row["agent"]: row for row in rows}
    assert by_agent["relation_agent"]["gated_r1_score"] > 0.7
    for agent in ["random", "shortcut", "passive_memory"]:
        assert by_agent[agent]["gated_r1_score"] == 0.0

