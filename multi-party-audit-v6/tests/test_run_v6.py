from __future__ import annotations

from src.run_v6_audit import run


def test_v6_run_only_compliant_resolver_passes() -> None:
    rows = run("configs/v6_audit.yaml")
    by_resolver = {row["resolver"]: row for row in rows}
    assert by_resolver["compliant_audit_resolver"]["gated_v6_score"] > 0.9
    for resolver in [
        "majority_vote_resolver",
        "confidence_only_resolver",
        "auto_compromise_resolver",
        "ignore_minority_risk_resolver",
        "no_audit_trail_resolver",
    ]:
        assert by_resolver[resolver]["gated_v6_score"] == 0.0

