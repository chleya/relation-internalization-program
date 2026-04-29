from __future__ import annotations

from src.case_schema import load_cases
from src.metrics_v6 import score_resolution
from src.resolvers import make_resolver


def test_compliant_resolver_routes_human_resolution() -> None:
    case = load_cases("cases/audit_cases.json")[0]
    record = make_resolver("compliant_audit_resolver").resolve(case)
    scores = score_resolution(case, record)
    assert scores["disagreement_detection"] == 1.0
    assert scores["no_auto_resolution"] == 1.0
    assert scores["human_resolution_routing"] == 1.0


def test_majority_vote_resolver_auto_resolves_and_fails() -> None:
    case = load_cases("cases/audit_cases.json")[0]
    record = make_resolver("majority_vote_resolver").resolve(case)
    scores = score_resolution(case, record)
    assert record["auto_resolved"] is True
    assert scores["no_auto_resolution"] == 0.0

