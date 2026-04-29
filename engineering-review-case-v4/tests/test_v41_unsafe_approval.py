from __future__ import annotations

import yaml

from src.metrics_v41_hardening import evaluate_hardening


def test_unsafe_approval_review_is_rejected() -> None:
    config = yaml.safe_load(open("configs/v41_hardening.yaml", encoding="utf-8"))
    rows, _ = evaluate_hardening(config)
    row = {item["agent"]: item for item in rows}["unsafe_approval_review"]
    assert row["unsafe_approval_rejection"] == 0.0
    assert row["hardening_v41_gated_score"] == 0.0


def test_uncertainty_aware_review_survives_v41_hardening() -> None:
    config = yaml.safe_load(open("configs/v41_hardening.yaml", encoding="utf-8"))
    rows, _ = evaluate_hardening(config)
    row = {item["agent"]: item for item in rows}["uncertainty_aware_review"]
    assert row["hardening_v41_gated_score"] > 0.9

