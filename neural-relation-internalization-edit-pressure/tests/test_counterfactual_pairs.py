from __future__ import annotations

from src.data import make_counterfactual_pair, make_record


def test_nuisance_counterfactual_preserves_label() -> None:
    record = make_record({"texture": "A", "wet": "dry", "color": "red", "odor": "strong"})
    left, right = make_counterfactual_pair(record, "nuisance_change")
    assert left["resource"] == right["resource"]
    assert left["color"] != right["color"]
    assert left["odor"] != right["odor"]


def test_relation_counterfactual_changes_relation_variables() -> None:
    record = make_record({"texture": "A", "wet": "dry", "color": "red", "odor": "strong"})
    _, right = make_counterfactual_pair(record, "relation_change")
    assert right["wet"] == "wet"
    assert right["resource"] == "poison"
