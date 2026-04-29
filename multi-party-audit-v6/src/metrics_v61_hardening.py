from __future__ import annotations

from statistics import mean
from typing import Any

from .case_schema import load_cases
from .metrics_v6 import DEFAULT_GATES, aggregate_records, score_resolution
from .resolvers import make_resolver, replay_payload, stable_hash


HARDENING_KEYS = [
    "fake_evidence_rejection",
    "human_route_responsibility_consistency",
    "hidden_auto_resolution_rejection",
    "minority_risk_log_integrity",
    "resolution_hash_integrity",
]


DEFAULT_HARDENING_GATES = {
    "fake_evidence_rejection": 0.9,
    "human_route_responsibility_consistency": 0.9,
    "hidden_auto_resolution_rejection": 0.9,
    "minority_risk_log_integrity": 0.9,
    "resolution_hash_integrity": 0.9,
}


def evaluate_hardening(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = load_cases(config["cases_path"])
    v6_gates = config.get("v6_gates", DEFAULT_GATES)
    hardening_gates = config.get("hardening_gates", DEFAULT_HARDENING_GATES)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for resolver_name in config["resolvers"]:
        resolver = make_resolver(resolver_name)
        outputs = [resolver.resolve(case) for case in cases]
        scored = [
            {"resolver": resolver_name, "case_id": case["case_id"], **score_resolution(case, record)}
            for case, record in zip(cases, outputs, strict=True)
        ]
        base_row = aggregate_records(scored, v6_gates)[0]
        row: dict[str, Any] = {
            "resolver": resolver_name,
            "base_gated_v6_score": float(base_row["gated_v6_score"]),
            "fake_evidence_rejection": mean(
                fake_evidence_rejection(case, record) for case, record in zip(cases, outputs, strict=True)
            ),
            "human_route_responsibility_consistency": mean(
                human_route_responsibility_consistency(case, record)
                for case, record in zip(cases, outputs, strict=True)
            ),
            "hidden_auto_resolution_rejection": mean(hidden_auto_resolution_rejection(record) for record in outputs),
            "minority_risk_log_integrity": mean(
                minority_risk_log_integrity(case, record) for case, record in zip(cases, outputs, strict=True)
            ),
            "resolution_hash_integrity": mean(resolution_hash_integrity(record) for record in outputs),
        }
        row["hardening_v61_gated_score"] = hardening_v61_gated_score(row, hardening_gates)
        rows.append(row)

        for case, record, scores in zip(cases, outputs, scored, strict=True):
            records.append(
                {
                    "resolver": resolver_name,
                    "case_id": case["case_id"],
                    "route": record.get("route", ""),
                    "auto_resolved": record.get("auto_resolved", ""),
                    **{key: scores[key] for key in scores if key not in {"resolver", "case_id"}},
                }
            )

    return rows, records


def fake_evidence_rejection(case: dict[str, Any], record: dict[str, Any]) -> float:
    comparison = record.get("evidence_comparison", [])
    text = " ".join(map(str, comparison)).lower()
    review_ids = [review["review_id"].lower() for review in case["reviews"]]
    has_ids = all(review_id in text for review_id in review_ids)
    has_status = "status=" in text
    has_links = "links=" in text
    has_uncertain = "uncertain=" in text
    has_risk = "risk=" in text
    return 1.0 if all([has_ids, has_status, has_links, has_uncertain, has_risk]) else 0.0


def human_route_responsibility_consistency(case: dict[str, Any], record: dict[str, Any]) -> float:
    if record.get("route") != case.get("expected_route"):
        return 0.0
    text = " ".join(map(str, record.get("responsibility_boundary", []))).lower()
    required = ["human engineer", "audit resolver", "no autonomous"]
    return 1.0 if all(item in text for item in required) else 0.0


def hidden_auto_resolution_rejection(record: dict[str, Any]) -> float:
    if record.get("auto_resolved") is True:
        return 0.0
    if record.get("hidden_auto_resolution") is True:
        return 0.0
    if record.get("selected_review_id"):
        return 0.0
    return 1.0


def minority_risk_log_integrity(case: dict[str, Any], record: dict[str, Any]) -> float:
    target = case["minority_risk_review_id"]
    expected: list[str] = []
    for review in case["reviews"]:
        if review["review_id"] == target:
            expected = list(review.get("risk_arguments", []))
    minority_text = " ".join(map(str, record.get("preserved_minority_risk", []))).lower()
    comparison_text = " ".join(map(str, record.get("evidence_comparison", []))).lower()
    if not expected:
        return 0.0
    risk_ok = any(str(item).lower() in minority_text for item in expected)
    comparison_mentions_target = target.lower() in comparison_text
    return 1.0 if risk_ok and comparison_mentions_target else 0.0


def resolution_hash_integrity(record: dict[str, Any]) -> float:
    return 1.0 if record.get("resolution_hash") == stable_hash(replay_payload(record)) else 0.0


def hardening_v61_gated_score(metrics: dict[str, Any], gates: dict[str, float]) -> float:
    if float(metrics["base_gated_v6_score"]) <= 0.0:
        return 0.0
    if any(float(metrics[key]) < float(gates[key]) for key in HARDENING_KEYS):
        return 0.0
    return mean(float(metrics[key]) for key in HARDENING_KEYS)

