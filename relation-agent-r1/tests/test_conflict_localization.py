from __future__ import annotations

from src.partial_agents import RelationSpecificUncertaintyAgent
from src.partial_metrics import conflict_cases, conflict_localization_accuracy


def test_conflict_audit_localizes_specific_relation_links() -> None:
    agent = RelationSpecificUncertaintyAgent()
    for state, expected_link in conflict_cases():
        decision = agent.decide(state)
        links = {item.get("link") for item in decision.audit}
        assert decision.action == "inspect"
        assert expected_link in links
        assert "state incomplete" not in " ".join(str(item) for item in decision.audit).lower()


def test_conflict_localization_metric_passes_for_relation_specific_agent() -> None:
    assert conflict_localization_accuracy("relation_specific_uncertainty_agent") >= 0.75
    assert conflict_localization_accuracy("missing_always_inspect") == 0.0
