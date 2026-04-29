from __future__ import annotations

from src.failure_localization import extraction_subspace_correlation, gate_failure_matrix, subspace_drop_by_seed


def sample_records() -> list[dict[str, str]]:
    return [
        {
            "model": "edit_pressure_training",
            "seed": "0",
            "ood_accuracy": "1.0",
            "shortcut_rejection_accuracy": "1.0",
            "reversal_adaptation_accuracy": "1.0",
            "counterfactual_consistency": "1.0",
            "table_alignment": "1.0",
            "table_ood_accuracy": "1.0",
            "table_spurious_attack_accuracy": "1.0",
            "table_edit_success": "1.0",
            "edit_locality": "1.0",
            "relation_subspace_drop": "0.1",
            "nuisance_subspace_drop": "0.0",
            "gated_internalization_score": "0.0",
        },
        {
            "model": "edit_pressure_training",
            "seed": "1",
            "ood_accuracy": "1.0",
            "shortcut_rejection_accuracy": "1.0",
            "reversal_adaptation_accuracy": "1.0",
            "counterfactual_consistency": "1.0",
            "table_alignment": "1.0",
            "table_ood_accuracy": "1.0",
            "table_spurious_attack_accuracy": "1.0",
            "table_edit_success": "1.0",
            "edit_locality": "1.0",
            "relation_subspace_drop": "0.3",
            "nuisance_subspace_drop": "0.0",
            "gated_internalization_score": "0.9",
        },
    ]


def test_gate_failure_matrix_identifies_relation_drop_failure() -> None:
    gates = {
        "ood_accuracy": 0.85,
        "shortcut_rejection_accuracy": 0.85,
        "reversal_adaptation_accuracy": 0.8,
        "counterfactual_consistency": 0.85,
        "table_alignment": 0.85,
        "table_edit_success": 0.9,
        "edit_locality": 0.85,
        "relation_subspace_drop": 0.2,
        "nuisance_subspace_drop_max": 0.1,
    }
    rows = gate_failure_matrix(sample_records(), gates)
    assert rows[0]["relation_subspace_drop_pass"] is False
    assert rows[1]["all_gates_pass"] is True


def test_subspace_drop_by_seed_extracts_edit_pressure_rows() -> None:
    rows = subspace_drop_by_seed(sample_records())
    assert rows[0]["relation_subspace_drop"] == 0.1
    assert rows[1]["gated_internalization_score"] == 0.9


def test_extraction_subspace_correlation_handles_constant_extraction_metrics() -> None:
    rows = extraction_subspace_correlation(sample_records())
    assert rows
    assert all("relation_subspace_drop_correlation" in row for row in rows)
