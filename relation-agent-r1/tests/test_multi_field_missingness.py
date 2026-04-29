from __future__ import annotations

from src.partial_agents import ActiveInspectionAgent
from src.partial_metrics import r3_multi_field_cases


def test_r3_cases_include_multiple_missing_fields() -> None:
    for _, observed, _ in r3_multi_field_cases():
        missing = [field for field, value in observed.items() if value == "unknown"]
        assert len(missing) >= 2


def test_active_agent_chooses_an_inspection_target_under_multi_field_missingness() -> None:
    agent = ActiveInspectionAgent()
    for _, observed, _ in r3_multi_field_cases():
        decision = agent.decide(observed)
        assert decision.action == "inspect"
        assert decision.inspect_field in {field for field, value in observed.items() if value == "unknown"}
