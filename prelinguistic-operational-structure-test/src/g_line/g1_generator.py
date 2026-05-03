from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeneratedRule:
    direct_threshold: float
    indirect_threshold: float
    inspect_threshold: float
    risk_threshold: float
    update_weight: float
    use_feedback: bool
    use_compression: bool

    @property
    def complexity(self) -> int:
        return 4 + int(self.use_feedback) + int(self.use_compression)


def fit_generator(model_inputs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("g1", {})
    candidates = generate_candidate_rules(section)
    best_rule = max(candidates, key=lambda rule: objective(rule, model_inputs, section))
    return {
        "rule": best_rule,
        "search_size": len(candidates),
        "train_objective": objective(best_rule, model_inputs, section),
        "generated_components": ["actionability_mask", "update_mask", "trace_revision_rule"],
    }


def generate_mask(model_input: dict[str, Any], artifact: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rule: GeneratedRule = artifact["rule"]
    mask = {}
    for item in model_input.get("interaction_history", []):
        region = int(item["region_id"])
        direct_score = direct_value(item, rule)
        indirect_score = indirect_value(item, rule)
        inspect_score = inspect_value(item, rule)
        risk = float(item["risk_proxy"])
        mask[region] = {
            "region_id": region,
            "inspectable": inspect_score >= rule.inspect_threshold,
            "directly_intervenable": direct_score >= rule.direct_threshold and risk <= rule.risk_threshold,
            "indirectly_intervenable": indirect_score >= rule.indirect_threshold and risk <= rule.risk_threshold + 0.10,
            "update_weight": rule.update_weight,
            "generated_trace_score": generated_trace_score(item, rule),
            "generated_by": "g1_compact_rule_search",
        }
    return mask


def choose_action(model_input: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    mask = generate_mask(model_input, artifact)
    best_region = None
    best_score = -1.0
    best_type = "abstain"
    for item in model_input.get("interaction_history", []):
        region = int(item["region_id"])
        info = mask[region]
        if info["directly_intervenable"]:
            score = float(item["intervention_gain"]) - float(item["risk_proxy"])
            if score > best_score:
                best_region, best_score, best_type = region, score, "apply_local_damping"
        if info["indirectly_intervenable"]:
            score = float(item["indirect_evidence"]) - 0.5 * float(item["risk_proxy"])
            if score > best_score:
                best_region, best_score, best_type = region, score, "indirect_stabilize"
    action = None if best_region is None else {"action_type": best_type, "region_id": best_region}
    return {
        "action": action,
        "generated_mask": mask,
        "generator_rule": artifact["rule"],
        "provenance": {"uses_evaluator_labels": False, "uses_oracle": False},
    }


def objective(rule: GeneratedRule, model_inputs: list[dict[str, Any]], section: dict[str, Any]) -> float:
    if not model_inputs:
        return -1.0
    compression_penalty = float(section.get("compression_penalty", 0.025)) * rule.complexity
    rewards = []
    for model_input in model_inputs:
        for item in model_input.get("interaction_history", []):
            direct = direct_value(item, rule)
            indirect = indirect_value(item, rule)
            inspect = inspect_value(item, rule)
            risk = float(item["risk_proxy"])
            reward = 0.0
            if direct >= rule.direct_threshold and risk <= rule.risk_threshold:
                reward = max(reward, float(item["observed_direct_reward"]))
            if indirect >= rule.indirect_threshold and risk <= rule.risk_threshold + 0.10:
                reward = max(reward, float(item["observed_indirect_reward"]))
            if inspect >= rule.inspect_threshold:
                reward = max(reward, 0.45 * float(item["observed_inspect_value"]))
            if reward == 0.0:
                reward = 0.08
            rewards.append(reward - compression_penalty)
    return sum(rewards) / len(rewards)


def generate_candidate_rules(section: dict[str, Any]) -> list[GeneratedRule]:
    thresholds = section.get("threshold_grid", [0.45, 0.55, 0.65, 0.75])
    risk_thresholds = section.get("risk_threshold_grid", [0.42, 0.52, 0.62])
    weights = section.get("update_weight_grid", [0.35, 0.55, 0.75])
    candidates = []
    for direct in thresholds:
        for indirect in thresholds:
            for inspect in thresholds:
                for risk in risk_thresholds:
                    for weight in weights:
                        for use_feedback in [False, True]:
                            for use_compression in [False, True]:
                                candidates.append(
                                    GeneratedRule(
                                        direct_threshold=float(direct),
                                        indirect_threshold=float(indirect),
                                        inspect_threshold=float(inspect),
                                        risk_threshold=float(risk),
                                        update_weight=float(weight),
                                        use_feedback=use_feedback,
                                        use_compression=use_compression,
                                    )
                                )
    return candidates


def direct_value(item: dict[str, Any], rule: GeneratedRule) -> float:
    value = 0.55 * float(item["intervention_gain"]) + 0.30 * float(item["prediction_error"])
    if rule.use_feedback:
        value += 0.15 * float(item["feedback_success"])
    return value


def indirect_value(item: dict[str, Any], rule: GeneratedRule) -> float:
    value = 0.75 * float(item["indirect_evidence"]) + 0.15 * float(item["feedback_success"])
    if rule.use_compression:
        value += 0.10 * float(item["compression_surprise"])
    return value


def inspect_value(item: dict[str, Any], rule: GeneratedRule) -> float:
    value = 0.60 * float(item["prediction_error"]) + 0.20 * float(item["delay_signal"])
    if rule.use_compression:
        value += 0.20 * float(item["compression_surprise"])
    return value


def generated_trace_score(item: dict[str, Any], rule: GeneratedRule) -> float:
    base = 0.5 * float(item["prediction_error"]) + 0.5 * float(item["feedback_success"])
    return (1.0 - rule.update_weight) * float(item["compression_surprise"]) + rule.update_weight * base
