from __future__ import annotations

import yaml

from src.metrics_v41_hardening import evaluate_hardening


def test_case_order_memory_fails_shuffled_order_attack() -> None:
    config = yaml.safe_load(open("configs/v41_hardening.yaml", encoding="utf-8"))
    rows, _ = evaluate_hardening(config)
    row = {item["agent"]: item for item in rows}["case_order_memory_review"]
    assert row["base_gated_v4_score"] > 0.0
    assert row["case_order_robustness"] < 0.8
    assert row["hardening_v41_gated_score"] == 0.0

