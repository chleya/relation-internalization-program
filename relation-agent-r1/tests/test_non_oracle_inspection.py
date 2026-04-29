from __future__ import annotations

from src.partial_env import PartialProcessWorld, hard_case_state, mask_fields
from src.partial_metrics import non_oracle_inspection_update_accuracy


def test_inspection_reveals_only_selected_field() -> None:
    env = PartialProcessWorld(seed=0, inspection_noise=0.0)
    true_state = hard_case_state()
    observed = mask_fields(true_state, ["pore_pressure", "displacement", "risk"])
    updated, result = env.inspect(true_state, observed, "pore_pressure")

    assert result.field == "pore_pressure"
    assert updated["pore_pressure"] == true_state["pore_pressure"]
    assert updated["displacement"] == "unknown"
    assert updated["risk"] == "unknown"


def test_non_oracle_update_accuracy_metric() -> None:
    config = {
        "inspection_noise": 0.10,
        "non_oracle_trials": 80,
    }
    assert non_oracle_inspection_update_accuracy("relation_specific_uncertainty_agent", 0, config) >= 0.70
