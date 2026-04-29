from __future__ import annotations

from src.case_schema import load_cases
from src.metrics_v6 import minority_risk_preservation
from src.resolvers import make_resolver


def test_compliant_resolver_preserves_minority_risk() -> None:
    case = load_cases("cases/audit_cases.json")[0]
    record = make_resolver("compliant_audit_resolver").resolve(case)
    assert minority_risk_preservation(case, record) == 1.0


def test_ignore_minority_risk_resolver_fails_minority_metric() -> None:
    case = load_cases("cases/audit_cases.json")[0]
    record = make_resolver("ignore_minority_risk_resolver").resolve(case)
    assert minority_risk_preservation(case, record) == 0.0

