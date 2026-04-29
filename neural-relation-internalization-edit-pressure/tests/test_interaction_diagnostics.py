from __future__ import annotations

from src.data import generate_dataset
from src.interaction_diagnostics import (
    binding_cases,
    edit_episode_dataset,
    edit_episode_pairs,
    evaluate_edit_state_swap,
    evaluate_query_support_binding,
    evaluate_support_shuffle_drop,
    run_multisite_ablation_matrix,
)
from src.models import make_model


def test_support_shuffle_drop_function_runs():
    model = make_model("edit_pressure_training", hidden_dim=8, bottleneck_dim=4)
    result = evaluate_support_shuffle_drop(model, edit_episode_dataset(seed=0, n=4), seed=0)
    assert {"normal_accuracy", "shuffled_support_accuracy", "support_shuffle_drop", "not_applicable"}.issubset(result.keys())


def test_edit_state_swap_returns_required_metrics():
    model = make_model("edit_pressure_training", hidden_dim=8, bottleneck_dim=4)
    result = evaluate_edit_state_swap(model, edit_episode_pairs(seed=1), seed=1)
    assert {"normal_edit_accuracy", "swapped_edit_effect_rate", "swapped_edit_locality", "edit_state_swap_success"}.issubset(result.keys())


def test_query_support_binding_returns_required_metrics():
    model = make_model("edit_pressure_training", hidden_dim=8, bottleneck_dim=4)
    result = evaluate_query_support_binding(model, binding_cases(seed=2), seed=2)
    assert {"support_conditioned_accuracy", "binding_sensitivity", "support_invariance_failure_rate"}.issubset(result.keys())


def test_ablation_matrix_has_required_columns():
    model = make_model("pure_prediction", hidden_dim=8, bottleneck_dim=4)
    rows = run_multisite_ablation_matrix(model, {"eval": generate_dataset(12, seed=3)}, seed=3)
    required = {"model", "seed", "site", "intervention_type", "base_accuracy", "intervened_accuracy", "accuracy_drop", "applicable"}
    assert rows
    assert required.issubset(rows[0].keys())
