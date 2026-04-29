from __future__ import annotations

import yaml

from src.metrics_v61_hardening import evaluate_hardening


def _rows() -> dict[str, dict[str, object]]:
    config = yaml.safe_load(open("configs/v61_hardening.yaml", encoding="utf-8"))
    rows, _ = evaluate_hardening(config)
    return {row["resolver"]: row for row in rows}


def test_compliant_resolver_survives_v61_hardening() -> None:
    row = _rows()["compliant_audit_resolver"]
    assert row["hardening_v61_gated_score"] > 0.9


def test_fake_evidence_comparison_resolver_is_rejected() -> None:
    row = _rows()["fake_evidence_comparison_resolver"]
    assert row["fake_evidence_rejection"] == 0.0
    assert row["hardening_v61_gated_score"] == 0.0


def test_hidden_auto_resolution_resolver_is_rejected() -> None:
    row = _rows()["hidden_auto_resolution_resolver"]
    assert row["hidden_auto_resolution_rejection"] == 0.0
    assert row["hardening_v61_gated_score"] == 0.0


def test_disagreement_without_minority_risk_is_rejected() -> None:
    row = _rows()["disagreement_logged_no_minority_resolver"]
    assert row["minority_risk_log_integrity"] == 0.0
    assert row["hardening_v61_gated_score"] == 0.0


def test_tampered_resolution_hash_is_rejected() -> None:
    row = _rows()["tampered_resolution_hash_resolver"]
    assert row["resolution_hash_integrity"] == 0.0
    assert row["hardening_v61_gated_score"] == 0.0

