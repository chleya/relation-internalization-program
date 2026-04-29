from __future__ import annotations

import yaml

from src.metrics_v51_hardening import evaluate_hardening


def _rows() -> dict[str, dict[str, object]]:
    config = yaml.safe_load(open("configs/v51_hardening.yaml", encoding="utf-8"))
    rows, _ = evaluate_hardening(config)
    return {row["shell"]: row for row in rows}


def test_compliant_shell_survives_v51_hardening() -> None:
    row = _rows()["compliant_shell"]
    assert row["hardening_v51_gated_score"] > 0.9


def test_fake_replay_shell_is_rejected() -> None:
    row = _rows()["fake_replay_shell"]
    assert row["fake_replay_rejection"] == 0.0
    assert row["hardening_v51_gated_score"] == 0.0


def test_gate_label_only_shell_is_rejected() -> None:
    row = _rows()["gate_label_only_shell"]
    assert row["gate_label_enforcement"] == 0.0
    assert row["hardening_v51_gated_score"] == 0.0


def test_missing_relation_evidence_shell_is_rejected() -> None:
    row = _rows()["missing_relation_evidence_shell"]
    assert row["relation_evidence_preservation"] < 0.9
    assert row["hardening_v51_gated_score"] == 0.0


def test_route_tampering_shell_is_rejected() -> None:
    row = _rows()["route_tampering_shell"]
    assert row["route_tamper_rejection"] == 0.0
    assert row["hardening_v51_gated_score"] == 0.0

