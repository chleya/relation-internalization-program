from __future__ import annotations

from src.partial_agents import MissingAlwaysInspectAgent, RelationSpecificUncertaintyAgent
from src.partial_env import NONCRITICAL_FIELDS, hard_case_state, mask_fields
from src.partial_metrics import noncritical_missing_no_inspect_rate


def test_noncritical_missingness_does_not_force_relation_specific_inspection() -> None:
    agent = RelationSpecificUncertaintyAgent()
    state = hard_case_state()
    for field in NONCRITICAL_FIELDS:
        decision = agent.decide(mask_fields(state, [field]))
        assert decision.action != "inspect"
        assert decision.inspect_field is None


def test_missing_always_inspect_baseline_inspects_noncritical_missingness() -> None:
    agent = MissingAlwaysInspectAgent()
    decision = agent.decide(mask_fields(hard_case_state(), ["contractor_report"]))
    assert decision.action == "inspect"
    assert decision.inspect_field == "contractor_report"


def test_noncritical_missing_no_inspect_metric_separates_agents() -> None:
    assert noncritical_missing_no_inspect_rate("relation_specific_uncertainty_agent") >= 0.8
    assert noncritical_missing_no_inspect_rate("missing_always_inspect") == 0.0
