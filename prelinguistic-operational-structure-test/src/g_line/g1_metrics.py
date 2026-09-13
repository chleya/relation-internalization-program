from __future__ import annotations

from typing import Any


SUMMARY_FIELDS = [
    "condition",
    "policy_name",
    "sample_count",
    "action_utility",
    "mask_f1",
    "compression_cost",
    "generator_score",
    "ood_generalization_score",
    "gain_over_random",
    "gain_over_hand_designed",
    "oracle_gap",
    "invalid_metric_count",
]

RECORD_FIELDS = [
    *SUMMARY_FIELDS,
    "episode_id",
    "action_type",
    "region_id",
    "action_score",
    "direct_mask_precision",
    "direct_mask_recall",
    "indirect_mask_precision",
    "indirect_mask_recall",
]


def score_episode(episode: dict[str, Any], output: dict[str, Any], policy_name: str) -> dict[str, Any]:
    truth = episode["evaluator_ground_truth"]["regions"]
    truth_by_region = {int(row["region_id"]): row for row in truth}
    action = output.get("action")
    if action is None:
        action_score = 0.40 if not any(row["direct_actionable"] or row["indirect_actionable"] for row in truth) else 0.20
        action_type = "abstain"
        region_id = ""
    else:
        region_id = int(action["region_id"])
        action_type = str(action["action_type"])
        region_truth = truth_by_region[region_id]
        if action_type == "apply_local_damping":
            action_score = 1.0 if region_truth["direct_actionable"] else (-0.25 if region_truth["latent_risk"] > 0.70 else 0.20)
        elif action_type == "indirect_stabilize":
            action_score = 0.90 if region_truth["indirect_actionable"] else 0.10
        else:
            action_score = 0.0
    mask = output.get("generated_mask", {})
    direct_stats = mask_stats(mask, truth_by_region, "directly_intervenable", "direct_actionable")
    indirect_stats = mask_stats(mask, truth_by_region, "indirectly_intervenable", "indirect_actionable")
    mask_f1 = (direct_stats["f1"] + indirect_stats["f1"]) / 2.0 if mask else 0.0
    compression_cost = compression_cost_for(output)
    generator_score = max(0.0, min(1.0, 0.55 * max(0.0, action_score) + 0.35 * mask_f1 + 0.10 * (1.0 - compression_cost)))
    return {
        "episode_id": int(episode["episode_id"]),
        "policy_name": policy_name,
        "condition": episode["condition"],
        "action_type": action_type,
        "region_id": region_id,
        "action_score": action_score,
        "action_utility": max(0.0, action_score),
        "mask_f1": mask_f1,
        "compression_cost": compression_cost,
        "generator_score": generator_score,
        "direct_mask_precision": direct_stats["precision"],
        "direct_mask_recall": direct_stats["recall"],
        "indirect_mask_precision": indirect_stats["precision"],
        "indirect_mask_recall": indirect_stats["recall"],
    }


def summarize(records: list[dict[str, Any]], baselines: dict[str, float]) -> dict[str, Any]:
    condition = str(records[0]["condition"]) if records else "missing"
    policy_name = str(records[0]["policy_name"]) if records else "missing"
    score = mean(records, "generator_score")
    oracle = baselines.get("oracle", 1.0)
    random = baselines.get("random", 0.0)
    hand = baselines.get("hand_designed", 0.0)
    return {
        "condition": condition,
        "policy_name": policy_name,
        "sample_count": len(records),
        "action_utility": mean(records, "action_utility"),
        "mask_f1": mean(records, "mask_f1"),
        "compression_cost": mean(records, "compression_cost"),
        "generator_score": score,
        "ood_generalization_score": score if condition == "ood_remap" else 0.0,
        "gain_over_random": score - random,
        "gain_over_hand_designed": score - hand,
        "oracle_gap": oracle - score,
        "invalid_metric_count": 1 if not records else 0,
    }


def mask_stats(mask: dict[int, dict[str, Any]], truth: dict[int, dict[str, Any]], pred_key: str, truth_key: str) -> dict[str, float]:
    tp = fp = fn = 0
    for region, row in truth.items():
        pred = bool(mask.get(region, {}).get(pred_key, False))
        actual = bool(row[truth_key])
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif (not pred) and actual:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compression_cost_for(output: dict[str, Any]) -> float:
    rule = output.get("generator_rule")
    if rule is None:
        return 0.0
    return min(1.0, float(rule.complexity) / 8.0)


def mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in records) / len(records)
