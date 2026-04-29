from __future__ import annotations

from src.partial_agents import RelationSpecificUncertaintyAgent
from src.partial_env import hard_case_state, mask_fields
from src.partial_metrics import mixed_observability_cases


def test_observed_high_risk_is_sufficient_despite_unknown_upstream_fields() -> None:
    agent = RelationSpecificUncertaintyAgent()
    state = mask_fields(hard_case_state(), ["pore_pressure", "displacement"])
    state["risk"] = "high"
    decision = agent.decide(state)
    assert decision.action == "stop_work"
    assert decision.inspect_field is None


def test_mixed_observability_cases_do_not_force_inspection() -> None:
    agent = RelationSpecificUncertaintyAgent()
    decisions = [agent.decide(case) for case in mixed_observability_cases()]
    assert all(decision.action != "inspect" for decision in decisions)
