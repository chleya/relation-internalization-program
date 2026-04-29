import numpy as np


def compute_average_reward(records: list[dict]) -> float:
    return float(np.mean([r["reward"] for r in records])) if records else 0.0


def compute_success_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    successes = 0
    for record in records:
        action = record["action"]
        resource = record["resource"]
        if (action == "eat" and resource == "food") or (action == "avoid" and resource in {"poison", "neutral"}):
            successes += 1
    return successes / len(records)


def adaptation_steps(rewards: list[float], threshold: float, window: int = 20) -> int:
    if len(rewards) < window:
        return len(rewards)
    for idx in range(window, len(rewards) + 1):
        if float(np.mean(rewards[idx - window : idx])) >= threshold:
            return idx
    return len(rewards)


def edit_success(before_actions: list[str], after_actions: list[str], target_contexts: list[dict]) -> float:
    if not target_contexts:
        return 0.0
    changed_to_avoid = sum(1 for before, after in zip(before_actions, after_actions) if before == "eat" and after == "avoid")
    return changed_to_avoid / len(target_contexts)


def edit_locality(before_actions: list[str], after_actions: list[str], non_target_contexts: list[dict]) -> float:
    if not non_target_contexts:
        return 1.0
    unchanged = sum(1 for before, after in zip(before_actions, after_actions) if before == after)
    return unchanged / len(non_target_contexts)


def internalization_score(metrics: dict) -> float:
    ood = metrics.get("ood_success", 0.0)
    spurious = metrics.get("spurious_robustness", 0.0)
    cf = metrics.get("counterfactual_accuracy", 0.0)
    edit = metrics.get("edit_success", 0.0)
    edit_resource = metrics.get("edit_resource_success", 0.0)
    edit_reversal = metrics.get("edit_reversal_success", 0.0)
    shuffle_drop = metrics.get("relation_shuffle_drop", 0.0)
    table_alignment = metrics.get("relation_table_alignment", 0.0)
    locality = metrics.get("edit_locality", 0.0)
    post_reward = max(0.0, min(1.0, (metrics.get("reversal_post_reward", 0.0) + 0.2) / 1.2))
    steps = metrics.get("reversal_adaptation_steps", 300)
    max_steps = metrics.get("reversal_post_steps", 300)
    adaptation = 1.0 - min(steps, max_steps) / max(max_steps, 1)
    return float(
        np.mean([ood, spurious, cf, edit, edit_resource, edit_reversal, shuffle_drop, table_alignment, locality, post_reward, adaptation])
    )


def gated_internalization_score(metrics: dict) -> float:
    gates = {
        "ood_success": 0.9,
        "spurious_resource_accuracy": 0.8,
        "counterfactual_accuracy": 0.9,
        "edit_resource_success": 0.9,
        "edit_reversal_success": 0.9,
        "relation_table_alignment": 0.9,
        "relation_shuffle_drop": 0.25,
    }
    for key, threshold in gates.items():
        if metrics.get(key, 0.0) < threshold:
            return 0.0
    return internalization_score(metrics)
