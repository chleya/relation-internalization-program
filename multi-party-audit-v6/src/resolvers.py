from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class BaseResolver:
    name: str

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class CompliantAuditResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("compliant_audit_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        record = base_record(case, self.name)
        record.update(
            {
                "material_disagreement_detected": material_disagreement(case),
                "evidence_comparison": compare_evidence(case),
                "preserved_minority_risk": minority_risk(case),
                "route": case["expected_route"],
                "auto_resolved": False,
                "responsibility_boundary": [
                    "human engineer resolves material disagreement",
                    "audit resolver preserves disagreement evidence",
                    "no autonomous engineering approval",
                ],
            }
        )
        record["resolution_hash"] = stable_hash(replay_payload(record))
        return record


class MajorityVoteResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("majority_vote_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        statuses = [review["status"] for review in case["reviews"]]
        status = Counter(statuses).most_common(1)[0][0]
        record = base_record(case, self.name)
        record.update(
            {
                "material_disagreement_detected": len(set(statuses)) > 1,
                "evidence_comparison": [],
                "preserved_minority_risk": [],
                "route": f"auto_{status}",
                "auto_resolved": True,
                "responsibility_boundary": ["majority vote selected outcome"],
            }
        )
        record["resolution_hash"] = stable_hash(replay_payload(record))
        return record


class ConfidenceOnlyResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("confidence_only_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        chosen = max(case["reviews"], key=lambda review: float(review["confidence"]))
        record = base_record(case, self.name)
        record.update(
            {
                "material_disagreement_detected": False,
                "evidence_comparison": [f"selected {chosen['review_id']} by confidence"],
                "preserved_minority_risk": [],
                "route": f"auto_{chosen['status']}",
                "auto_resolved": True,
                "responsibility_boundary": ["confidence score selected outcome"],
            }
        )
        record["resolution_hash"] = stable_hash(replay_payload(record))
        return record


class AutoCompromiseResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("auto_compromise_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        record = base_record(case, self.name)
        record.update(
            {
                "material_disagreement_detected": True,
                "evidence_comparison": ["merged competing views into compromise"],
                "preserved_minority_risk": [],
                "route": "auto_compromise",
                "auto_resolved": True,
                "responsibility_boundary": ["compromise selected automatically"],
            }
        )
        record["resolution_hash"] = stable_hash(replay_payload(record))
        return record


class IgnoreMinorityRiskResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("ignore_minority_risk_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        record = base_record(case, self.name)
        record.update(
            {
                "material_disagreement_detected": True,
                "evidence_comparison": compare_evidence(case)[:1],
                "preserved_minority_risk": [],
                "route": case["expected_route"],
                "auto_resolved": False,
                "responsibility_boundary": ["human engineer remains responsible"],
            }
        )
        record["resolution_hash"] = stable_hash(replay_payload(record))
        return record


class NoAuditTrailResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("no_audit_trail_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "resolver": self.name,
            "route": case["expected_route"],
            "auto_resolved": False,
        }


class FakeEvidenceComparisonResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("fake_evidence_comparison_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        record = CompliantAuditResolver().resolve(case)
        record["resolver"] = self.name
        record["evidence_comparison"] = [
            "all reviews were compared and the evidence was considered"
        ]
        record["resolution_hash"] = stable_hash(replay_payload(record))
        return record


class HumanRouteLabelOnlyResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("human_route_label_only_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        record = CompliantAuditResolver().resolve(case)
        record["resolver"] = self.name
        record["route"] = "human_resolution_required"
        record["responsibility_boundary"] = ["resolved by process label"]
        record["resolution_hash"] = stable_hash(replay_payload(record))
        return record


class HiddenAutoResolutionResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("hidden_auto_resolution_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        record = CompliantAuditResolver().resolve(case)
        record["resolver"] = self.name
        record["selected_review_id"] = case["reviews"][0]["review_id"]
        record["hidden_auto_resolution"] = True
        record["resolution_hash"] = stable_hash(replay_payload(record))
        return record


class DisagreementLoggedNoMinorityResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("disagreement_logged_no_minority_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        record = CompliantAuditResolver().resolve(case)
        record["resolver"] = self.name
        record["preserved_minority_risk"] = ["minority risk noted"]
        record["resolution_hash"] = stable_hash(replay_payload(record))
        return record


class TamperedResolutionHashResolver(BaseResolver):
    def __init__(self) -> None:
        super().__init__("tampered_resolution_hash_resolver")

    def resolve(self, case: dict[str, Any]) -> dict[str, Any]:
        record = CompliantAuditResolver().resolve(case)
        record["resolver"] = self.name
        original_hash = record["resolution_hash"]
        record["route"] = "auto_compromise"
        record["auto_resolved"] = True
        record["resolution_hash"] = original_hash
        return record


def base_record(case: dict[str, Any], resolver: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "resolver": resolver,
        "review_ids": [review["review_id"] for review in case["reviews"]],
        "review_statuses": [review["status"] for review in case["reviews"]],
        "expected_route": case["expected_route"],
        "expected_material_disagreement": case["expected_material_disagreement"],
        "minority_risk_review_id": case["minority_risk_review_id"],
    }


def material_disagreement(case: dict[str, Any]) -> bool:
    statuses = {review["status"] for review in case["reviews"]}
    uncertain_sets = {tuple(review.get("uncertain_links", [])) for review in case["reviews"]}
    chains = {tuple(review.get("relation_chain", [])) for review in case["reviews"]}
    return len(statuses) > 1 or len(uncertain_sets) > 1 or len(chains) > 1


def compare_evidence(case: dict[str, Any]) -> list[str]:
    comparison = []
    for review in case["reviews"]:
        comparison.append(
            f"{review['review_id']}: status={review['status']}; "
            f"links={len(review.get('relation_chain', []))}; "
            f"uncertain={len(review.get('uncertain_links', []))}; "
            f"risk={'; '.join(review.get('risk_arguments', []))}"
        )
    return comparison


def minority_risk(case: dict[str, Any]) -> list[str]:
    target = case["minority_risk_review_id"]
    for review in case["reviews"]:
        if review["review_id"] == target:
            return list(review.get("risk_arguments", []))
    return []


def replay_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "resolver": record.get("resolver"),
        "review_ids": record.get("review_ids", []),
        "review_statuses": record.get("review_statuses", []),
        "material_disagreement_detected": record.get("material_disagreement_detected"),
        "evidence_comparison": record.get("evidence_comparison", []),
        "preserved_minority_risk": record.get("preserved_minority_risk", []),
        "route": record.get("route"),
        "auto_resolved": record.get("auto_resolved"),
        "responsibility_boundary": record.get("responsibility_boundary", []),
    }


def make_resolver(name: str) -> BaseResolver:
    resolvers = {
        "compliant_audit_resolver": CompliantAuditResolver,
        "majority_vote_resolver": MajorityVoteResolver,
        "confidence_only_resolver": ConfidenceOnlyResolver,
        "auto_compromise_resolver": AutoCompromiseResolver,
        "ignore_minority_risk_resolver": IgnoreMinorityRiskResolver,
        "no_audit_trail_resolver": NoAuditTrailResolver,
        "fake_evidence_comparison_resolver": FakeEvidenceComparisonResolver,
        "human_route_label_only_resolver": HumanRouteLabelOnlyResolver,
        "hidden_auto_resolution_resolver": HiddenAutoResolutionResolver,
        "disagreement_logged_no_minority_resolver": DisagreementLoggedNoMinorityResolver,
        "tampered_resolution_hash_resolver": TamperedResolutionHashResolver,
    }
    if name not in resolvers:
        raise ValueError(f"unknown resolver: {name}")
    return resolvers[name]()
