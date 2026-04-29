from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .partial_env import CRITICAL_FIELDS, NONCRITICAL_FIELDS


@dataclass(frozen=True)
class AgentDecision:
    action: str
    inspect_field: str | None = None
    audit: list[dict[str, Any]] = field(default_factory=list)
    inspection_value: dict[str, float] = field(default_factory=dict)


LINK_BY_FIELD = {
    "pore_pressure": "RainfallDrainage -> PorePressure",
    "anchoring": "Anchoring -> DisplacementDown",
    "displacement": "PorePressure -> Displacement",
    "crack": "Displacement -> Crack",
    "risk": "Crack -> Risk",
}


class PartialAgent:
    name = "partial_agent"

    def decide(self, state: dict[str, str]) -> AgentDecision:
        raise NotImplementedError


class MissingAlwaysInspectAgent(PartialAgent):
    name = "missing_always_inspect"

    def decide(self, state: dict[str, str]) -> AgentDecision:
        for field, value in state.items():
            if value == "unknown":
                return AgentDecision(
                    action="inspect",
                    inspect_field=field,
                    audit=[{"reason": "any_missing_field", "field": field}],
                )
        return AgentDecision(action=choose_action_from_observed_state(state))


class FirstMissingInspectAgent(PartialAgent):
    name = "first_missing_inspect"

    def decide(self, state: dict[str, str]) -> AgentDecision:
        for field, value in state.items():
            if value == "unknown":
                return AgentDecision(
                    action="inspect",
                    inspect_field=field,
                    audit=[{"reason": "first_missing_field", "field": field}],
                    inspection_value={field: 1.0},
                )
        return AgentDecision(action=choose_action_from_observed_state(state))


class RandomInspectFieldAgent(PartialAgent):
    name = "random_inspect_field"

    def decide(self, state: dict[str, str]) -> AgentDecision:
        missing = [field for field, value in state.items() if value == "unknown"]
        if missing:
            field = sorted(missing)[0]
            return AgentDecision(
                action="inspect",
                inspect_field=field,
                audit=[{"reason": "deterministic_random_baseline", "field": field}],
                inspection_value={field: 1.0},
            )
        return AgentDecision(action=choose_action_from_observed_state(state))


class RiskFirstInspectAgent(PartialAgent):
    name = "risk_first_inspect"

    def decide(self, state: dict[str, str]) -> AgentDecision:
        missing = [field for field, value in state.items() if value == "unknown"]
        if "risk" in missing:
            return AgentDecision(action="inspect", inspect_field="risk", inspection_value={"risk": 1.0})
        if missing:
            field = missing[0]
            return AgentDecision(action="inspect", inspect_field=field, inspection_value={field: 1.0})
        return AgentDecision(action=choose_action_from_observed_state(state))


class RelationSpecificUncertaintyAgent(PartialAgent):
    name = "relation_specific_uncertainty_agent"

    def decide(self, state: dict[str, str]) -> AgentDecision:
        audit = relation_audit(state)
        conflicts = [item for item in audit if item["reason"] == "conflicting_relation"]
        if conflicts:
            return AgentDecision(action="inspect", inspect_field=conflicts[0]["inspect_field"], audit=audit)

        if observed_sufficient_action(state) is not None:
            return AgentDecision(action=observed_sufficient_action(state), audit=audit)

        critical_missing = [
            field
            for field in CRITICAL_FIELDS
            if state.get(field) == "unknown" and field_is_relevant_to_unresolved_chain(field, state)
        ]
        if critical_missing:
            field = critical_missing[0]
            return AgentDecision(action="inspect", inspect_field=field, audit=audit or [missing_audit_item(field)])

        return AgentDecision(action=choose_action_from_observed_state(state), audit=audit)


class ActiveInspectionAgent(PartialAgent):
    name = "active_inspection_agent"

    def decide(self, state: dict[str, str]) -> AgentDecision:
        audit = relation_audit(state)
        conflicts = [item for item in audit if item["reason"] == "conflicting_relation"]
        values = inspection_values(state)

        if conflicts:
            field = max((item["inspect_field"] for item in conflicts), key=lambda candidate: values.get(candidate, 0.0))
            return AgentDecision(action="inspect", inspect_field=field, audit=audit, inspection_value=values)

        sufficient = observed_sufficient_action(state)
        if sufficient is not None:
            return AgentDecision(action=sufficient, audit=audit, inspection_value=values)

        missing = [field for field, value in state.items() if value == "unknown"]
        if missing:
            field = max(missing, key=lambda candidate: (values.get(candidate, 0.0), candidate))
            if values.get(field, 0.0) > 0.0:
                return AgentDecision(action="inspect", inspect_field=field, audit=audit, inspection_value=values)

        return AgentDecision(action=choose_action_from_observed_state(state), audit=audit, inspection_value=values)


def make_partial_agent(name: str) -> PartialAgent:
    if name == "active_inspection_agent":
        return ActiveInspectionAgent()
    if name == "relation_specific_uncertainty_agent":
        return RelationSpecificUncertaintyAgent()
    if name == "missing_always_inspect":
        return MissingAlwaysInspectAgent()
    if name == "first_missing_inspect":
        return FirstMissingInspectAgent()
    if name == "random_inspect_field":
        return RandomInspectFieldAgent()
    if name == "risk_first_inspect":
        return RiskFirstInspectAgent()
    raise ValueError(f"unknown partial agent: {name}")


def observed_sufficient_action(state: dict[str, str]) -> str | None:
    if state.get("risk") == "high":
        return "stop_work"
    if state.get("risk") == "low":
        return "monitor"
    if state.get("crack") == "open":
        return "stop_work"
    if state.get("displacement") == "high":
        return "stop_work"
    if state.get("pore_pressure") == "high" and state.get("anchoring") == "absent":
        return "stop_work"
    return None


def choose_action_from_observed_state(state: dict[str, str]) -> str:
    sufficient = observed_sufficient_action(state)
    if sufficient is not None:
        return sufficient
    if state.get("pore_pressure") == "high" and state.get("drainage") == "poor":
        return "improve_drainage"
    if state.get("pore_pressure") == "high" and state.get("anchoring") == "absent":
        return "add_anchoring"
    return "monitor"


def field_is_relevant_to_unresolved_chain(field: str, state: dict[str, str]) -> bool:
    if field in NONCRITICAL_FIELDS:
        return False
    if state.get("risk") in {"high", "low"}:
        return False
    if state.get("crack") in {"open", "closed"} and field in {"pore_pressure", "anchoring", "displacement"}:
        return False
    if state.get("displacement") in {"high", "normal"} and field in {"pore_pressure", "anchoring"}:
        return False
    return field in CRITICAL_FIELDS


def relation_audit(state: dict[str, str]) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for field in CRITICAL_FIELDS:
        if state.get(field) == "unknown" and field_is_relevant_to_unresolved_chain(field, state):
            audit.append(missing_audit_item(field))

    if state.get("pore_pressure") == "normal" and state.get("anchoring") == "absent" and state.get("displacement") == "high":
        audit.append(conflict_item("PorePressure -> Displacement", "displacement"))
    if state.get("anchoring") == "present" and state.get("displacement") == "high":
        audit.append(conflict_item("Anchoring -> DisplacementDown", "displacement"))
    if state.get("displacement") == "normal" and state.get("crack") == "open":
        audit.append(conflict_item("Displacement -> Crack", "crack"))
    if state.get("displacement") == "high" and state.get("crack") == "closed":
        audit.append(conflict_item("Displacement -> Crack", "crack"))
    return audit


def missing_audit_item(field: str) -> dict[str, Any]:
    return {
        "reason": "missing_relation_evidence",
        "field": field,
        "link": LINK_BY_FIELD[field],
        "inspect_field": field,
    }


def conflict_item(link: str, inspect_field: str) -> dict[str, Any]:
    return {
        "reason": "conflicting_relation",
        "link": link,
        "inspect_field": inspect_field,
    }


def inspection_values(state: dict[str, str]) -> dict[str, float]:
    values = {}
    for field, current in state.items():
        if current != "unknown":
            continue
        chain_position = {
            "pore_pressure": 1.0,
            "anchoring": 1.5,
            "displacement": 2.0,
            "crack": 2.5,
            "risk": 3.0,
        }.get(field, 0.0)
        action_relevance = 2.0 if field in {"risk", "crack", "displacement"} else 1.0 if field in {"pore_pressure", "anchoring"} else 0.0
        conflict_resolution = 1.0 if any(item.get("inspect_field") == field for item in relation_audit({**state, field: "normal"})) else 0.0
        noncritical_penalty = 3.0 if field in NONCRITICAL_FIELDS else 0.0
        values[field] = chain_position + action_relevance + conflict_resolution - noncritical_penalty
    return values
