from __future__ import annotations

from src.data import generate_dataset, resource_rule


def test_base_resource_rule() -> None:
    assert resource_rule({"texture": "A", "wet": "dry"}) == "food"
    assert resource_rule({"texture": "A", "wet": "wet"}) == "poison"
    assert resource_rule({"texture": "B", "wet": "dry"}) == "poison"
    assert resource_rule({"texture": "C", "wet": "wet"}) == "neutral"


def test_dataset_generation_includes_labels() -> None:
    rows = generate_dataset(10, seed=0, regime="base", shortcut_mode="train")
    assert len(rows) == 10
    assert {"texture", "wet", "color", "odor", "resource", "action"} <= set(rows[0])
