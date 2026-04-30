from __future__ import annotations

from src.substrate_audit import substrate_profiles


def test_substrate_audit_records_injected_priors() -> None:
    rows = {row["model"]: row for row in substrate_profiles()}
    assert rows["pixel_predictor"]["language_or_relation_table_input"] is False
    assert rows["predictive_coding_model"]["explicit_object_prior"] is False
    assert rows["patch_graph_model"]["prior_level"] == "medium_grid_prior"
    assert rows["koopman_model"]["prior_level"] == "medium_linear_dynamics_prior"
    assert rows["slot_model"]["explicit_object_prior"] is True
    assert rows["field_model"]["explicit_field_prior"] is True
    assert rows["flow_checkpoint_model"]["prior_level"] == "high_checkpoint_prior"
    assert rows["schema_model"]["schema_head_prior"] is True
