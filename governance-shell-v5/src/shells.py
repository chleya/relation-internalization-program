from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class BaseShell:
    name: str

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class CompliantShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("compliant_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        event = governance_event(case, shell=self.name)
        event["route"] = case["expected_route"]
        event["auto_approved"] = False
        event["approval_gate"] = "blocked_auto_approval"
        event["responsibility_chain"] = [
            "AI reviewer produced bounded diagnostic",
            "governance shell blocked autonomous approval",
            "human engineer retains decision authority",
        ]
        event["non_deployment_boundary"] = [
            "toy governance diagnostic only",
            "not a real approval workflow",
        ]
        event["replay_hash"] = stable_hash(replay_payload(event))
        return event


class AutoApproveShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("auto_approve_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        event = governance_event(case, shell=self.name)
        event["route"] = "auto_approved"
        event["auto_approved"] = True
        event["approval_gate"] = "passed_without_human"
        event["responsibility_chain"] = ["AI approved case"]
        event["non_deployment_boundary"] = []
        event["replay_hash"] = stable_hash(replay_payload(event))
        return event


class NoLogShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("no_log_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "shell": self.name,
            "route": case["expected_route"],
            "auto_approved": False,
        }


class NoReplayShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("no_replay_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        event = governance_event(case, shell=self.name)
        event["route"] = case["expected_route"]
        event["auto_approved"] = False
        event["approval_gate"] = "blocked_auto_approval"
        event["responsibility_chain"] = ["human engineer remains responsible"]
        event["non_deployment_boundary"] = ["toy governance diagnostic only"]
        event["replay_hash"] = ""
        return event


class NoResponsibilityShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("no_responsibility_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        event = governance_event(case, shell=self.name)
        event["route"] = case["expected_route"]
        event["auto_approved"] = False
        event["approval_gate"] = "blocked_auto_approval"
        event["responsibility_chain"] = []
        event["non_deployment_boundary"] = ["toy governance diagnostic only"]
        event["replay_hash"] = stable_hash(replay_payload(event))
        return event


class FakeReplayShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("fake_replay_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        event = CompliantShell().process(case)
        event["shell"] = self.name
        event["replay_hash"] = "fake-replay-hash"
        return event


class GateLabelOnlyShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("gate_label_only_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        event = governance_event(case, shell=self.name)
        event["route"] = "auto_approved"
        event["auto_approved"] = True
        event["approval_gate"] = "blocked_auto_approval"
        event["responsibility_chain"] = [
            "AI reviewer produced bounded diagnostic",
            "governance shell claims a blocked gate",
            "human engineer remains responsible",
        ]
        event["non_deployment_boundary"] = ["toy governance diagnostic only", "not a real approval workflow"]
        event["replay_hash"] = stable_hash(replay_payload(event))
        return event


class ResponsibilityBoilerplateShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("responsibility_boilerplate_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        event = governance_event(case, shell=self.name)
        event["route"] = "auto_approved"
        event["auto_approved"] = True
        event["approval_gate"] = "passed_without_human"
        event["responsibility_chain"] = [
            "human engineer remains responsible",
            "human engineer remains responsible",
            "human engineer remains responsible",
        ]
        event["non_deployment_boundary"] = ["toy governance diagnostic only"]
        event["replay_hash"] = stable_hash(replay_payload(event))
        return event


class MissingRelationEvidenceShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("missing_relation_evidence_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        event = CompliantShell().process(case)
        event["shell"] = self.name
        event["relation_chain"] = []
        event["uncertain_links"] = []
        event["replay_hash"] = stable_hash(replay_payload(event))
        return event


class RouteTamperingShell(BaseShell):
    def __init__(self) -> None:
        super().__init__("route_tampering_shell")

    def process(self, case: dict[str, Any]) -> dict[str, Any]:
        event = CompliantShell().process(case)
        event["shell"] = self.name
        original_hash = stable_hash(replay_payload(event))
        event["route"] = "auto_approved"
        event["auto_approved"] = True
        event["replay_hash"] = original_hash
        return event


def governance_event(case: dict[str, Any], shell: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "shell": shell,
        "review_status": case["review_status"],
        "relation_chain": list(case.get("relation_chain", [])),
        "uncertain_links": list(case.get("uncertain_links", [])),
        "takeover_conditions": list(case.get("takeover_conditions", [])),
        "verification_indicators": list(case.get("verification_indicators", [])),
        "claim_boundary": list(case.get("claim_boundary", [])),
        "expected_route": case["expected_route"],
        "expected_auto_approval": case["expected_auto_approval"],
    }


def replay_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": event.get("case_id"),
        "review_status": event.get("review_status"),
        "relation_chain": event.get("relation_chain", []),
        "uncertain_links": event.get("uncertain_links", []),
        "takeover_conditions": event.get("takeover_conditions", []),
        "verification_indicators": event.get("verification_indicators", []),
        "route": event.get("route"),
        "auto_approved": event.get("auto_approved"),
        "approval_gate": event.get("approval_gate"),
        "responsibility_chain": event.get("responsibility_chain", []),
        "non_deployment_boundary": event.get("non_deployment_boundary", []),
    }


def make_shell(name: str) -> BaseShell:
    shells = {
        "compliant_shell": CompliantShell,
        "auto_approve_shell": AutoApproveShell,
        "no_log_shell": NoLogShell,
        "no_replay_shell": NoReplayShell,
        "no_responsibility_shell": NoResponsibilityShell,
        "fake_replay_shell": FakeReplayShell,
        "gate_label_only_shell": GateLabelOnlyShell,
        "responsibility_boilerplate_shell": ResponsibilityBoilerplateShell,
        "missing_relation_evidence_shell": MissingRelationEvidenceShell,
        "route_tampering_shell": RouteTamperingShell,
    }
    if name not in shells:
        raise ValueError(f"unknown shell: {name}")
    return shells[name]()
