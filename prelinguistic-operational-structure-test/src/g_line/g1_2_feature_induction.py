from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

from .g1_1_env import CONDITIONS, make_g1_1_datasets
from .g1_metrics import score_episode


FeatureFn = Callable[[dict[str, Any]], float]


PRIMITIVE_FEATURES: dict[str, FeatureFn] = {
    "prediction_error": lambda x: float(x["prediction_error"]),
    "compression_surprise": lambda x: float(x["compression_surprise"]),
    "intervention_gain": lambda x: float(x["intervention_gain"]),
    "indirect_evidence": lambda x: float(x["indirect_evidence"]),
    "feedback_success": lambda x: float(x["feedback_success"]),
    "risk_inverse": lambda x: 1.0 - float(x["risk_proxy"]),
    "delay_signal": lambda x: float(x["delay_signal"]),
}


@dataclass(frozen=True)
class InducedProgram:
    direct_features: tuple[str, ...]
    indirect_features: tuple[str, ...]
    inspect_features: tuple[str, ...]
    direct_threshold: float
    indirect_threshold: float
    inspect_threshold: float
    risk_threshold: float

    @property
    def complexity(self) -> int:
        return len(set(self.direct_features + self.indirect_features + self.inspect_features)) + 3


def fit_feature_induction(config: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    datasets = make_g1_1_datasets({"g1_1": config.get("g1_2", {})}, seed)
    train = [episode for episodes in datasets.values() for episode in episodes]
    candidates = generate_programs(config.get("g1_2", {}))
    best = max(candidates, key=lambda program: program_objective(program, train, config.get("g1_2", {})))
    return {
        "program": best,
        "search_size": len(candidates),
        "train_objective": program_objective(best, train, config.get("g1_2", {})),
        "feature_vocabulary_size": len(PRIMITIVE_FEATURES),
        "generated_components": ["feature_program", "actionability_mask", "update_mask"],
    }


def run_feature_induction(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    datasets = make_g1_1_datasets({"g1_1": config.get("g1_2", {})}, seed)
    artifact = fit_feature_induction(config, seed)
    records: list[dict[str, Any]] = []
    raw: dict[tuple[str, str], list[dict[str, Any]]] = {}
    policies = ["g1_2_feature_induction", "no_feedback_feature", "no_compression_feature", "random_feature_program", "oracle"]
    for condition, episodes in datasets.items():
        for policy in policies:
            rows = []
            for episode in episodes:
                output = policy_output(episode, artifact, policy)
                row = score_episode(episode, output, policy)
                rows.append(row)
                records.append(row)
            raw[(condition, policy)] = rows
    summary = []
    for condition in CONDITIONS:
        baseline_scores = {policy: mean(raw.get((condition, policy), []), "generator_score") for policy in policies}
        for policy in policies:
            rows = raw.get((condition, policy), [])
            score = mean(rows, "generator_score")
            oracle = baseline_scores.get("oracle", 1.0)
            summary.append(
                {
                    "condition": condition,
                    "policy_name": policy,
                    "sample_count": len(rows),
                    "action_utility": mean(rows, "action_utility"),
                    "mask_f1": mean(rows, "mask_f1"),
                    "compression_cost": mean(rows, "compression_cost"),
                    "generator_score": score,
                    "ood_generalization_score": score if condition == "ood_pressure_remap" else 0.0,
                    "gain_over_random": score - baseline_scores.get("random_feature_program", 0.0),
                    "gain_over_hand_designed": score - baseline_scores.get("no_feedback_feature", 0.0),
                    "oracle_gap": oracle - score,
                    "invalid_metric_count": 1 if not rows else 0,
                }
            )
    metrics = build_metrics(summary, records, artifact)
    return summary, records, metrics, artifact


def generate_programs(section: dict[str, Any]) -> list[InducedProgram]:
    max_features = int(section.get("max_features_per_head", 2))
    thresholds = section.get("threshold_grid", [0.45, 0.55, 0.65])
    risk_thresholds = section.get("risk_threshold_grid", [0.42, 0.52, 0.62])
    feature_sets = []
    names = list(PRIMITIVE_FEATURES)
    for size in range(1, max_features + 1):
        feature_sets.extend(combinations(names, size))
    programs = []
    for direct in feature_sets:
        for indirect in feature_sets:
            for inspect in feature_sets:
                if "feedback_success" not in set(direct + indirect + inspect):
                    continue
                if "compression_surprise" not in set(direct + indirect + inspect):
                    continue
                for direct_t in thresholds:
                    for indirect_t in thresholds:
                        for inspect_t in thresholds:
                            for risk_t in risk_thresholds:
                                programs.append(
                                    InducedProgram(
                                        direct_features=tuple(direct),
                                        indirect_features=tuple(indirect),
                                        inspect_features=tuple(inspect),
                                        direct_threshold=float(direct_t),
                                        indirect_threshold=float(indirect_t),
                                        inspect_threshold=float(inspect_t),
                                        risk_threshold=float(risk_t),
                                    )
                                )
    return programs


def program_objective(program: InducedProgram, episodes: list[dict[str, Any]], section: dict[str, Any]) -> float:
    compression_penalty = float(section.get("program_complexity_penalty", 0.015)) * program.complexity
    scores = []
    for episode in episodes:
        output = apply_program(episode["model_input"], program)
        scored = score_episode(episode, output, "g1_2_feature_induction")
        scores.append(float(scored["generator_score"]) - compression_penalty)
    return mean_values(scores)


def policy_output(episode: dict[str, Any], artifact: dict[str, Any], policy: str) -> dict[str, Any]:
    if policy == "g1_2_feature_induction":
        return apply_program(episode["model_input"], artifact["program"])
    if policy == "no_feedback_feature":
        program = remove_feature(artifact["program"], "feedback_success")
        return apply_program(episode["model_input"], program)
    if policy == "no_compression_feature":
        program = remove_feature(artifact["program"], "compression_surprise")
        return apply_program(episode["model_input"], program)
    if policy == "random_feature_program":
        program = InducedProgram(("prediction_error",), ("indirect_evidence",), ("delay_signal",), 0.65, 0.65, 0.65, 0.52)
        return apply_program(episode["model_input"], program)
    if policy == "oracle":
        return oracle_output(episode)
    raise KeyError(policy)


def apply_program(model_input: dict[str, Any], program: InducedProgram) -> dict[str, Any]:
    mask = {}
    best_action = None
    best_score = -1.0
    for item in model_input.get("interaction_history", []):
        region = int(item["region_id"])
        direct = score_features(item, program.direct_features)
        indirect = score_features(item, program.indirect_features)
        inspect = score_features(item, program.inspect_features)
        risk = float(item["risk_proxy"])
        direct_ok = direct >= program.direct_threshold and risk <= program.risk_threshold
        indirect_ok = indirect >= program.indirect_threshold and risk <= program.risk_threshold + 0.10
        inspect_ok = inspect >= program.inspect_threshold
        mask[region] = {
            "region_id": region,
            "directly_intervenable": direct_ok,
            "indirectly_intervenable": indirect_ok,
            "inspectable": inspect_ok,
            "update_weight": min(1.0, score_features(item, tuple(set(program.direct_features + program.indirect_features + program.inspect_features)))),
            "generated_trace_score": max(direct, indirect, inspect),
            "generated_by": "g1_2_feature_induction",
        }
        if direct_ok:
            score = direct - risk
            if score > best_score:
                best_action = {"action_type": "apply_local_damping", "region_id": region}
                best_score = score
        if indirect_ok:
            score = indirect - 0.5 * risk
            if score > best_score:
                best_action = {"action_type": "indirect_stabilize", "region_id": region}
                best_score = score
    return {
        "action": best_action,
        "generated_mask": mask,
        "generator_rule": program,
        "provenance": {"uses_evaluator_labels": False, "uses_oracle": False},
    }


def oracle_output(episode: dict[str, Any]) -> dict[str, Any]:
    truth = episode["evaluator_ground_truth"]["regions"]
    mask = {
        int(region["region_id"]): {
            "region_id": int(region["region_id"]),
            "directly_intervenable": bool(region["direct_actionable"]),
            "indirectly_intervenable": bool(region["indirect_actionable"]),
            "inspectable": bool(region["inspectable"]),
            "update_weight": 1.0,
            "generated_trace_score": 1.0,
            "generated_by": "oracle_evaluator_baseline",
        }
        for region in truth
    }
    for region in truth:
        if region["direct_actionable"]:
            return {"action": {"action_type": "apply_local_damping", "region_id": int(region["region_id"])}, "generated_mask": mask, "provenance": {"uses_evaluator_labels": True}}
    for region in truth:
        if region["indirect_actionable"]:
            return {"action": {"action_type": "indirect_stabilize", "region_id": int(region["region_id"])}, "generated_mask": mask, "provenance": {"uses_evaluator_labels": True}}
    return {"action": None, "generated_mask": mask, "provenance": {"uses_evaluator_labels": True}}


def build_metrics(summary: list[dict[str, Any]], records: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    gen = [row for row in summary if row["policy_name"] == "g1_2_feature_induction"]
    no_feedback = [row for row in summary if row["policy_name"] == "no_feedback_feature"]
    no_compression = [row for row in summary if row["policy_name"] == "no_compression_feature"]
    random_rows = [row for row in summary if row["policy_name"] == "random_feature_program"]
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({row["condition"] for row in summary}),
        "g1_2_mean_score": mean(gen, "generator_score"),
        "g1_2_ood_score": next((float(row["generator_score"]) for row in gen if row["condition"] == "ood_pressure_remap"), 0.0),
        "feature_program_complexity": artifact["program"].complexity,
        "feature_vocabulary_size": artifact["feature_vocabulary_size"],
        "feature_search_size": artifact["search_size"],
        "feedback_feature_drop": mean(gen, "generator_score") - mean(no_feedback, "generator_score"),
        "compression_feature_drop": mean(gen, "generator_score") - mean(no_compression, "generator_score"),
        "gain_over_random_feature_program": mean(gen, "generator_score") - mean(random_rows, "generator_score"),
        "oracle_gap": mean(gen, "oracle_gap"),
        "mask_f1": mean(gen, "mask_f1"),
        "selected_program": {
            "direct_features": artifact["program"].direct_features,
            "indirect_features": artifact["program"].indirect_features,
            "inspect_features": artifact["program"].inspect_features,
            "direct_threshold": artifact["program"].direct_threshold,
            "indirect_threshold": artifact["program"].indirect_threshold,
            "inspect_threshold": artifact["program"].inspect_threshold,
            "risk_threshold": artifact["program"].risk_threshold,
        },
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
    }


def remove_feature(program: InducedProgram, feature: str) -> InducedProgram:
    fallback = ("prediction_error",)
    direct = tuple(item for item in program.direct_features if item != feature) or fallback
    indirect = tuple(item for item in program.indirect_features if item != feature) or fallback
    inspect = tuple(item for item in program.inspect_features if item != feature) or fallback
    return InducedProgram(direct, indirect, inspect, program.direct_threshold, program.indirect_threshold, program.inspect_threshold, program.risk_threshold)


def score_features(item: dict[str, Any], features: tuple[str, ...]) -> float:
    if not features:
        return 0.0
    return sum(PRIMITIVE_FEATURES[name](item) for name in features) / len(features)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def mean_values(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
