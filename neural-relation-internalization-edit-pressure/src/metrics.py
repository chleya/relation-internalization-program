from __future__ import annotations


def gated_internalization_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    required = [
        metrics["ood_accuracy"] >= gates["ood_accuracy"],
        metrics["shortcut_rejection_accuracy"] >= gates["shortcut_rejection_accuracy"],
        metrics["reversal_adaptation_accuracy"] >= gates["reversal_adaptation_accuracy"],
        metrics["counterfactual_consistency"] >= gates["counterfactual_consistency"],
        metrics["table_alignment"] >= gates["table_alignment"],
        metrics["table_edit_success"] >= gates["table_edit_success"],
        metrics["edit_locality"] >= gates["edit_locality"],
        metrics["relation_subspace_drop"] >= gates["relation_subspace_drop"],
        metrics["nuisance_subspace_drop"] <= gates["nuisance_subspace_drop_max"],
    ]
    if not all(required):
        return 0.0
    normalized_drop = min(1.0, metrics["relation_subspace_drop"] / 0.5)
    return (
        0.15 * metrics["ood_accuracy"]
        + 0.15 * metrics["shortcut_rejection_accuracy"]
        + 0.15 * metrics["reversal_adaptation_accuracy"]
        + 0.10 * metrics["counterfactual_consistency"]
        + 0.15 * metrics["table_alignment"]
        + 0.10 * metrics["table_edit_success"]
        + 0.10 * metrics["edit_locality"]
        + 0.10 * normalized_drop
    )
