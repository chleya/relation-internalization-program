from __future__ import annotations

import yaml

from src.metrics_v41_hardening import evaluate_hardening


def _rows() -> dict[str, dict[str, object]]:
    config = yaml.safe_load(open("configs/v41_hardening.yaml", encoding="utf-8"))
    rows, _ = evaluate_hardening(config)
    return {row["agent"]: row for row in rows}


def test_schema_template_review_is_rejected() -> None:
    row = _rows()["schema_template_review"]
    assert row["schema_template_rejection"] < 0.8
    assert row["hardening_v41_gated_score"] == 0.0


def test_fluent_nonspecific_review_is_rejected() -> None:
    row = _rows()["fluent_nonspecific_review"]
    assert row["fluent_nonspecific_rejection"] < 0.8
    assert row["hardening_v41_gated_score"] == 0.0

