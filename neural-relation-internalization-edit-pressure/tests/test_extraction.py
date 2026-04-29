from __future__ import annotations

from src.data import canonical_context
from src.extraction import extract_table_from_model, table_alignment
from src.models import ExplicitTableOracle


def test_extract_table_from_oracle_aligns_with_base_rules() -> None:
    table = extract_table_from_model(ExplicitTableOracle())
    assert table_alignment(table) == 1.0
    assert table.infer_resource(canonical_context("A", "dry")) == "food"


def test_extracted_table_edit_is_local() -> None:
    table = extract_table_from_model(ExplicitTableOracle())
    before = table.infer_resource(canonical_context("A", "wet"))
    assert table.edit_rule({"texture": "A", "wet": "dry"}, "poison")
    assert table.infer_resource(canonical_context("A", "dry")) == "poison"
    assert table.infer_resource(canonical_context("A", "wet")) == before
