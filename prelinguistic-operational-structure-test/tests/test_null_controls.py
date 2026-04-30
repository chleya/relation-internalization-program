from __future__ import annotations

from src.models import make_model
from src.null_controls import evaluate_null_controls

from .conftest import small_config


def test_null_controls_return_expected_keys() -> None:
    metrics = evaluate_null_controls(make_model("schema_model"), small_config(), seed=10)
    assert "null_event_activation" in metrics
    assert "null_inspection_concentration" in metrics
    assert "null_prior_alarm" in metrics


def test_slot_model_has_null_inspection_prior_alarm() -> None:
    metrics = evaluate_null_controls(make_model("slot_model"), small_config(), seed=11)
    assert metrics["null_inspection_concentration"] > 0.25
    assert metrics["null_prior_alarm"] > 0.0
