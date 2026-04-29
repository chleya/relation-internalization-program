from __future__ import annotations

from src.partial_agents import inspection_values
from src.partial_env import hard_case_state, mask_fields


def test_inspection_value_penalizes_noncritical_fields() -> None:
    state = mask_fields(hard_case_state(), ["contractor_report", "risk"])
    values = inspection_values(state)
    assert values["risk"] > values["contractor_report"]


def test_inspection_value_prefers_action_relevant_downstream_field() -> None:
    state = mask_fields(hard_case_state(), ["pore_pressure", "displacement", "risk"])
    values = inspection_values(state)
    assert values["risk"] > values["displacement"] > values["pore_pressure"]
