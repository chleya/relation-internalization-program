from __future__ import annotations

from typing import Any

from .g1_generator import GeneratedRule, direct_value, generate_candidate_rules, generate_mask, indirect_value, inspect_value


def fit_pressure_generator(model_inputs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("g1_1", {})
    candidates = generate_candidate_rules(section)
    best_rule = max(candidates, key=lambda rule: pressure_objective(rule, model_inputs, section))
    return {
        "rule": best_rule,
        "search_size": len(candidates),
        "train_objective": pressure_objective(best_rule, model_inputs, section),
        "generated_components": ["actionability_mask", "update_mask", "trace_revision_rule"],
    }


def choose_pressure_action(model_input: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    rule: GeneratedRule = artifact["rule"]
    mask = generate_mask(model_input, artifact)
    best_region = None
    best_type = "abstain"
    best_score = -1.0
    for item in model_input.get("interaction_history", []):
        region = int(item["region_id"])
        risk = float(item["risk_proxy"])
        direct_score = direct_value(item, rule) - 0.7 * risk
        indirect_score = indirect_value(item, rule) - 0.35 * risk
        inspect_score = inspect_value(item, rule)
        if mask[region]["directly_intervenable"] and direct_score > best_score:
            best_region, best_type, best_score = region, "apply_local_damping", direct_score
        if mask[region]["indirectly_intervenable"] and indirect_score > best_score:
            best_region, best_type, best_score = region, "indirect_stabilize", indirect_score
        if best_region is None and inspect_score >= rule.inspect_threshold:
            best_region, best_type, best_score = region, "inspect", inspect_score * 0.2
    action = None if best_region is None else {"action_type": best_type, "region_id": best_region}
    return {
        "action": action,
        "generated_mask": mask,
        "generator_rule": rule,
        "provenance": {"uses_evaluator_labels": False, "uses_oracle": False},
    }


def pressure_objective(rule: GeneratedRule, model_inputs: list[dict[str, Any]], section: dict[str, Any]) -> float:
    if not model_inputs:
        return -1.0
    compression_penalty = float(section.get("compression_penalty", 0.02)) * rule.complexity
    rewards = []
    for model_input in model_inputs:
        best = -0.05
        for item in model_input.get("interaction_history", []):
            risk = float(item["risk_proxy"])
            if direct_value(item, rule) >= rule.direct_threshold and risk <= rule.risk_threshold:
                best = max(best, float(item["observed_direct_reward"]))
            if indirect_value(item, rule) >= rule.indirect_threshold and risk <= rule.risk_threshold + 0.10:
                best = max(best, float(item["observed_indirect_reward"]))
            if inspect_value(item, rule) >= rule.inspect_threshold:
                best = max(best, 0.35 * float(item["observed_inspect_value"]))
        rewards.append(best - compression_penalty)
    return sum(rewards) / len(rewards)


def pressure_ablation_artifacts(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rule: GeneratedRule = artifact["rule"]
    variants = {
        "no_feedback": GeneratedRule(
            rule.direct_threshold,
            rule.indirect_threshold,
            rule.inspect_threshold,
            rule.risk_threshold,
            rule.update_weight,
            False,
            rule.use_compression,
        ),
        "no_compression": GeneratedRule(
            rule.direct_threshold,
            rule.indirect_threshold,
            rule.inspect_threshold,
            rule.risk_threshold,
            rule.update_weight,
            rule.use_feedback,
            False,
        ),
    }
    return {
        name: {**artifact, "rule": variant, "generated_components": artifact["generated_components"]}
        for name, variant in variants.items()
    }
