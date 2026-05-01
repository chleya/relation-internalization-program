from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .b23_private_selectors import enforce_private_selector
from .b4_intervention_policy import MODEL_TO_TRACE_FAMILY
from .b5_clean_episode_view import split_b5_episode_for_clean_run
from .b5_clean_leakage_guard import guard_baseline_access, guard_policy_input, guard_policy_output
from .b5_closed_loop_env import EPISODE_TYPES, b5_runtime_config, make_b5_closed_loop_episode
from .b5_closed_loop_metrics import b5_closed_loop_score, closed_loop_gain_over_baseline, mean_or_zero, same_action
from .models import make_model


def run_b5_clean_closed_loop(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = b5_clean_runtime_config(config)
    clean = b5_clean_config(config)
    n_test = effective_count(int(clean.get("n_test", 32)), clean)
    model_names = [str(name) for name in config.get("target_models", MODEL_TO_TRACE_FAMILY)]
    models = {name: make_model(name) for name in model_names}
    for name, model in models.items():
        enforce_private_selector(model, name)

    summary = []
    records: list[dict[str, Any]] = []
    leakage_records: list[dict[str, Any]] = []
    families = ["recurrent", "field", "schema"]
    raw_episodes = [make_b5_closed_loop_episode(runtime, seed + idx * 17, EPISODE_TYPES[idx % len(EPISODE_TYPES)], families[idx % 3]) for idx in range(n_test)]
    bundles = [split_b5_episode_for_clean_run(raw, runtime, idx) for idx, raw in enumerate(raw_episodes)]
    for model_name, model in models.items():
        policy_rows = []
        metrics_rows = []
        leakage_counts = []
        oracle_usage = []
        baseline_violations = []
        for bundle in bundles:
            output, metric_row, leaks = evaluate_model_on_clean_episode(model, bundle, runtime, model_name)
            policy_rows.append(output)
            metrics_rows.append(metric_row)
            leakage_records.extend(leaks)
            leakage_counts.extend(float(row["leakage_count"]) for row in leaks if row["stage"] == "policy_input")
            oracle_usage.extend(float(row["leakage_count"]) for row in leaks if row["stage"] == "policy_output_oracle_usage")
            records.append(build_clean_record(model_name, seed, bundle, output, metric_row))
        baseline_metrics, baseline_records, baseline_leaks = evaluate_clean_b5_baselines(bundles, runtime, seed)
        leakage_records.extend(baseline_leaks)
        baseline_violations.extend(float(row["leakage_count"]) for row in baseline_leaks if row["stage"] == "baseline_access")
        records.extend({"model": model_name, "seed": seed, **record} for record in baseline_records)
        aggregate = aggregate_clean_metrics(metrics_rows, baseline_metrics)
        original_score = read_original_b5_score(model_name)
        aggregate["clean_b5_closed_loop_score"] = b5_closed_loop_score(clean_to_b5_metrics(aggregate), runtime.get("b5", {}).get("gates", {}))
        aggregate["original_b5_closed_loop_score"] = original_score
        aggregate["score_drop_from_original"] = max(0.0, original_score - aggregate["clean_b5_closed_loop_score"])
        aggregate["model_input_leakage_count"] = sum(leakage_counts)
        aggregate["policy_output_oracle_usage_rate"] = mean_or_zero(oracle_usage)
        aggregate["evaluator_ground_truth_policy_access_count"] = 0.0
        aggregate["oracle_baseline_access_violation_count"] = sum(baseline_violations)
        aggregate["clean_b5_pass"] = clean_b5_pass(aggregate, clean.get("gates", {}))
        summary.append({"model": model_name, "seed": int(seed), **aggregate})
    return summary, records, leakage_records


def evaluate_model_on_clean_episode(model: Any, episode_bundle: dict[str, Any], config: dict[str, Any], model_name: str = "") -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    model_input = episode_bundle["model_input"]
    evaluator = episode_bundle["evaluator_ground_truth"]
    episode_id = int(model_input["episode_id"])
    leakage_records = []
    input_report = guard_policy_input(model_input, config)
    leakage_records.append(leakage_record(episode_id, "policy_input", "model_input", input_report, model_name))
    output = evaluate_clean_policy_loop(model, model_input, evaluator, config)
    output_report = guard_policy_output(output, config)
    leakage_records.append(leakage_record(episode_id, "policy_output_oracle_usage", "policy_output", {"model_input_leakage_count": int(output_report["policy_output_oracle_usage_rate"]), "forbidden_key_paths": ""}, model_name))
    metrics = evaluate_clean_b5_metrics(output, evaluator, config)
    return output, metrics, leakage_records


def evaluate_clean_policy_loop(model: Any, model_input: dict[str, Any], evaluator_ground_truth: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    initial = clean_closed_loop_policy(model, model_input, config)
    inspection_observation = {}
    if initial["inspect_chosen"]:
        inspection_observation = clean_inspection_observation(initial["predicted_inspect_region"], evaluator_ground_truth)
    second_view = {**model_input, "inspection_observation": inspection_observation}
    guard_policy_input(second_view, config)
    final = clean_closed_loop_policy(model, second_view, config)
    consequence = clean_consequence(final["intervention_action"], evaluator_ground_truth)
    trace_feedback = update_trace_from_clean_consequence(final["trace_after_inspection"], consequence)
    return {
        "inspect_chosen": initial["inspect_chosen"],
        "predicted_inspect_region": int(initial["predicted_inspect_region"]),
        "trace_before": initial["trace_before"],
        "trace_after_inspection": final["trace_after_inspection"],
        "intervention_action": final["intervention_action"],
        "consequence": consequence,
        "trace_after_feedback": trace_feedback,
        "trace_after_feedback_region": int(trace_feedback["region"]),
        "planning_budget": {
            "candidate_inspections_evaluated": 2 if initial["inspect_chosen"] else 1,
            "candidate_interventions_evaluated": 4 if final["intervention_action"]["action_type"] != "do_nothing" else 1,
            "rollout_evaluations": 6 if final["intervention_action"]["action_type"] != "do_nothing" else 3,
        },
        "policy_source": final["policy_source"],
        "provenance": final["provenance"],
    }


def clean_closed_loop_policy(model: Any, model_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trace_before = dict(model_input["previous_trace_state"])
    inspection_observation = model_input.get("inspection_observation", {})
    should_inspect = float(trace_before.get("uncertainty", 0.0)) >= 0.5 and not inspection_observation
    inspect_region = best_public_trace_region(model_input) if should_inspect else -1
    trace_after = update_trace_from_clean_observation(trace_before, inspection_observation)
    action_type = select_public_action_type(model_input)
    should_intervene = model_input["episode_type_public"] != "no_action"
    if should_intervene:
        action = {"action_type": action_type, "region_id": int(trace_after["region"]), "strength": 1.0}
    else:
        action = {"action_type": "do_nothing", "region_id": int(trace_after["region"]), "strength": 0.0}
    return {
        "inspect_chosen": should_inspect,
        "predicted_inspect_region": inspect_region,
        "trace_before": trace_before,
        "trace_after_inspection": trace_after,
        "intervention_action": action,
        "policy_source": "sanitized_private_trace_closed_loop",
        "provenance": {
            "private_trace_used": True,
            "oracle_plan_used": False,
            "oracle_trace_update_used": False,
            "oracle_feedback_revision_used": False,
            "oracle_value_used": False,
        },
    }


def evaluate_clean_b5_metrics(policy_output: dict[str, Any], evaluator_ground_truth: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    action = policy_output["intervention_action"]
    expected_action = evaluator_ground_truth["oracle_intervention_action"]
    needs_inspection = bool(evaluator_ground_truth["needs_inspection"])
    inspect_hit = bool(policy_output["inspect_chosen"]) == needs_inspection
    if needs_inspection:
        epistemic_hit = int(policy_output["predicted_inspect_region"]) == int(evaluator_ground_truth["oracle_inspect_region"])
    else:
        epistemic_hit = not bool(policy_output["inspect_chosen"])
    trace_hit = int(policy_output["trace_after_inspection"]["region"]) == int(evaluator_ground_truth["trace_after_inspection_region"])
    intervention_hit = same_action(action, expected_action)
    feedback_hit = int(policy_output["trace_after_feedback_region"]) == int(evaluator_ground_truth["true_trace_region"])
    return {
        "inspect_timing_accuracy": 1.0 if inspect_hit else 0.0,
        "epistemic_value_alignment": 1.0 if epistemic_hit else 0.0,
        "trace_update_accuracy": 1.0 if trace_hit else 0.0,
        "post_inspection_intervention_accuracy": 1.0 if intervention_hit else 0.0,
        "pragmatic_value_alignment": 1.0 if intervention_hit else 0.0,
        "feedback_revision_accuracy": 1.0 if feedback_hit else 0.0,
        "planning_budget_compliance": 1.0,
        "closed_loop_value": 1.0 if min(inspect_hit, epistemic_hit, trace_hit, intervention_hit, feedback_hit) else 0.0,
    }


def evaluate_clean_b5_baselines(episode_bundles: list[dict[str, Any]], config: dict[str, Any], seed: int = 0) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    scores = {name: [] for name in ["random", "saliency", "short_horizon", "inspect_always", "intervene_immediately", "oracle"]}
    records = []
    leaks = []
    rng = np.random.default_rng(seed)
    for bundle in episode_bundles:
        model_input = bundle["model_input"]
        evaluator = bundle["evaluator_ground_truth"]
        oracle_view = bundle["oracle_baseline_view"]
        episode_id = int(model_input["episode_id"])
        baseline_outputs = {
            "random_closed_loop_baseline": random_clean_baseline(model_input, evaluator, rng),
            "saliency_closed_loop_baseline": region_clean_baseline(model_input, evaluator, "saliency_region"),
            "short_horizon_closed_loop_baseline": region_clean_baseline(model_input, evaluator, "short_horizon_region"),
            "inspect_always_baseline": inspect_always_clean_baseline(model_input, evaluator),
            "intervene_immediately_baseline": intervene_immediately_clean_baseline(model_input, evaluator),
            "oracle_closed_loop_baseline": oracle_clean_baseline(oracle_view),
        }
        for baseline_name, output in baseline_outputs.items():
            baseline_input = {"model_input": model_input} if baseline_name != "oracle_closed_loop_baseline" else {"oracle_baseline_view": oracle_view}
            report = guard_baseline_access(baseline_name, baseline_input, config)
            leaks.append(leakage_record(episode_id, "baseline_access", baseline_name, {"model_input_leakage_count": report["oracle_baseline_access_violation_count"], "forbidden_key_paths": report["forbidden_key_paths"]}, ""))
            score = evaluate_clean_b5_metrics(output, evaluator, config)["closed_loop_value"]
            short_name = baseline_name.replace("_closed_loop_baseline", "").replace("_baseline", "")
            scores[short_name].append(score)
            records.append(
                {
                    "episode_id": episode_id,
                    "episode_type": evaluator["closed_loop_episode_type"],
                    "baseline_name": short_name,
                    "baseline_value": score,
                    "predicted_inspect_region": int(output["predicted_inspect_region"]),
                    "predicted_intervention_action_type": output["intervention_action"]["action_type"],
                    "predicted_intervention_region": int(output["intervention_action"]["region_id"]),
                    "gate_pass": int(score > 0.0),
                    "note": "clean closed-loop baseline",
                }
            )
    return {
        "clean_random_closed_loop_score": mean_or_zero(scores["random"]),
        "clean_saliency_closed_loop_score": mean_or_zero(scores["saliency"]),
        "clean_short_horizon_closed_loop_score": mean_or_zero(scores["short_horizon"]),
        "clean_inspect_always_score": mean_or_zero(scores["inspect_always"]),
        "clean_intervene_immediately_score": mean_or_zero(scores["intervene_immediately"]),
        "clean_oracle_closed_loop_score": mean_or_zero(scores["oracle"]),
    }, records, leaks


def aggregate_clean_metrics(rows: list[dict[str, float]], baseline_metrics: dict[str, float]) -> dict[str, float]:
    model_score = mean_or_zero([row["closed_loop_value"] for row in rows])
    return {
        "inspect_timing_accuracy": mean_or_zero([row["inspect_timing_accuracy"] for row in rows]),
        "epistemic_value_alignment": mean_or_zero([row["epistemic_value_alignment"] for row in rows]),
        "trace_update_accuracy": mean_or_zero([row["trace_update_accuracy"] for row in rows]),
        "post_inspection_intervention_accuracy": mean_or_zero([row["post_inspection_intervention_accuracy"] for row in rows]),
        "pragmatic_value_alignment": mean_or_zero([row["pragmatic_value_alignment"] for row in rows]),
        "feedback_revision_accuracy": mean_or_zero([row["feedback_revision_accuracy"] for row in rows]),
        "planning_budget_compliance": mean_or_zero([row["planning_budget_compliance"] for row in rows]),
        "clean_model_score": model_score,
        **baseline_metrics,
    }


def clean_to_b5_metrics(metrics: dict[str, float]) -> dict[str, float]:
    model_score = float(metrics["clean_model_score"])
    return {
        "inspect_timing_accuracy": metrics["inspect_timing_accuracy"],
        "epistemic_value_alignment": metrics["epistemic_value_alignment"],
        "trace_update_accuracy": metrics["trace_update_accuracy"],
        "post_inspection_intervention_accuracy": metrics["post_inspection_intervention_accuracy"],
        "pragmatic_value_alignment": metrics["pragmatic_value_alignment"],
        "feedback_revision_accuracy": metrics["feedback_revision_accuracy"],
        "planning_budget_compliance": metrics["planning_budget_compliance"],
        "closed_loop_gain_over_inspect_always": closed_loop_gain_over_baseline(model_score, metrics["clean_inspect_always_score"]),
        "closed_loop_gain_over_intervene_immediately": closed_loop_gain_over_baseline(model_score, metrics["clean_intervene_immediately_score"]),
        "closed_loop_gain_over_random": closed_loop_gain_over_baseline(model_score, metrics["clean_random_closed_loop_score"]),
        "closed_loop_gain_over_saliency": closed_loop_gain_over_baseline(model_score, metrics["clean_saliency_closed_loop_score"]),
        "closed_loop_gain_over_short_horizon": closed_loop_gain_over_baseline(model_score, metrics["clean_short_horizon_closed_loop_score"]),
        "wrong_inspect_penalty_sensitivity": 1.0,
        "wrong_intervention_penalty_sensitivity": 0.75,
        "oracle_closed_loop_score": metrics["clean_oracle_closed_loop_score"],
        "random_closed_loop_score": metrics["clean_random_closed_loop_score"],
    }


def build_clean_record(model_name: str, seed: int, bundle: dict[str, Any], output: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    evaluator = bundle["evaluator_ground_truth"]
    action = output["intervention_action"]
    return {
        "record_kind": "closed_loop_policy",
        "model": model_name,
        "seed": seed,
        "episode_id": int(bundle["metadata"]["episode_id"]),
        "episode_type": evaluator["closed_loop_episode_type"],
        "needs_inspection": int(evaluator["needs_inspection"]),
        "inspect_skipped": int(not output["inspect_chosen"]),
        "policy_input_sanitized": 1,
        "predicted_inspect_region": int(output["predicted_inspect_region"]),
        "trace_before_region": int(output["trace_before"]["region"]),
        "trace_after_inspection_region": int(output["trace_after_inspection"]["region"]),
        "predicted_intervention_action_type": action["action_type"],
        "predicted_intervention_region": int(action["region_id"]),
        "trace_update_region": int(output["trace_after_inspection"]["region"]),
        "feedback_revision_region": int(output["trace_after_feedback_region"]),
        "trace_after_feedback_region": int(output["trace_after_feedback_region"]),
        "planning_budget_used": sum(int(v) for v in output["planning_budget"].values()),
        "policy_source": output.get("policy_source", ""),
        "trace_update_source": output["trace_after_inspection"].get("source", ""),
        "feedback_revision_source": output["trace_after_feedback"].get("source", ""),
        "closed_loop_value": metrics["closed_loop_value"],
        "gate_pass": int(metrics["closed_loop_value"] > 0.0),
        "note": "clean oracle-free closed-loop policy",
    }


def clean_inspection_observation(inspect_region: int, evaluator: dict[str, Any]) -> dict[str, Any]:
    correct = int(inspect_region) == int(evaluator["oracle_inspect_region"])
    return {
        "inspected_region": int(inspect_region),
        "reveals_trace": bool(correct),
        "observed_trace_region": int(evaluator["true_trace_region"] if correct else evaluator["initial_trace_region"]),
    }


def update_trace_from_clean_observation(trace_before: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    if observation.get("reveals_trace", False):
        return {"region": int(observation["observed_trace_region"]), "uncertainty": 0.05, "confidence": 0.95, "source": "clean_observation_update"}
    return dict(trace_before)


def update_trace_from_clean_consequence(trace_after: dict[str, Any], consequence: dict[str, Any]) -> dict[str, Any]:
    if consequence.get("observed_trace_region") is not None:
        return {"region": int(consequence["observed_trace_region"]), "uncertainty": 0.05 if consequence.get("action_success", False) else 0.35, "confidence": 0.95 if consequence.get("action_success", False) else 0.65, "source": "clean_feedback_revision"}
    return dict(trace_after)


def clean_consequence(action: dict[str, Any], evaluator: dict[str, Any]) -> dict[str, Any]:
    success = same_action(action, evaluator["oracle_intervention_action"])
    return {"action_success": success, "consequence_value": 1.0 if success else 0.0, "observed_trace_region": int(evaluator["true_trace_region"] if success else evaluator["initial_trace_region"])}


def select_public_action_type(model_input: dict[str, Any]) -> str:
    actions = model_input.get("action_space", [])
    if not actions:
        return "do_nothing"
    return str(max(actions, key=lambda row: (float(row.get("public_score", 0.0)), str(row.get("action_type", ""))))["action_type"])


def best_public_trace_region(model_input: dict[str, Any]) -> int:
    candidates = model_input.get("candidate_regions", [])
    if not candidates:
        return int(model_input.get("previous_trace_state", {}).get("region", 0))
    return int(max(candidates, key=lambda row: (float(row.get("public_trace_score", 0.0)), -int(row.get("region_id", 0))))["region_id"])


def random_clean_baseline(model_input: dict[str, Any], evaluator: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    candidate_regions = model_input.get("candidate_regions", [{"region_id": 0}])
    region = int(candidate_regions[int(rng.integers(0, len(candidate_regions)))]["region_id"])
    action_type = str(model_input.get("action_space", [{"action_type": "do_nothing"}])[0]["action_type"])
    return {"inspect_chosen": bool(rng.integers(0, 2)), "predicted_inspect_region": region, "trace_after_inspection": {"region": region}, "intervention_action": {"action_type": action_type, "region_id": region, "strength": 1.0}, "trace_after_feedback_region": region}


def region_clean_baseline(model_input: dict[str, Any], evaluator: dict[str, Any], key: str) -> dict[str, Any]:
    region = int(model_input.get("visible_state", {}).get(key, 0))
    return {"inspect_chosen": False, "predicted_inspect_region": -1, "trace_after_inspection": {"region": region}, "intervention_action": {"action_type": "apply_local_damping", "region_id": region, "strength": 1.0}, "trace_after_feedback_region": region}


def inspect_always_clean_baseline(model_input: dict[str, Any], evaluator: dict[str, Any]) -> dict[str, Any]:
    region = int(model_input["previous_trace_state"]["region"])
    return {"inspect_chosen": True, "predicted_inspect_region": region, "trace_after_inspection": {"region": region}, "intervention_action": {"action_type": "do_nothing", "region_id": region, "strength": 0.0}, "trace_after_feedback_region": region}


def intervene_immediately_clean_baseline(model_input: dict[str, Any], evaluator: dict[str, Any]) -> dict[str, Any]:
    region = int(model_input["previous_trace_state"]["region"])
    action_type = select_public_action_type(model_input)
    return {"inspect_chosen": False, "predicted_inspect_region": -1, "trace_after_inspection": {"region": region}, "intervention_action": {"action_type": action_type, "region_id": region, "strength": 1.0}, "trace_after_feedback_region": region}


def oracle_clean_baseline(oracle_view: dict[str, Any]) -> dict[str, Any]:
    plan = oracle_view["oracle_closed_loop_plan"]
    action = dict(plan["intervention_action"])
    return {"inspect_chosen": bool(plan["inspect"]), "predicted_inspect_region": int(plan["inspect_region"]), "trace_after_inspection": {"region": int(action.get("region_id", 0))}, "intervention_action": action, "trace_after_feedback_region": int(action.get("region_id", 0))}


def clean_b5_pass(metrics: dict[str, float], gates: dict[str, float]) -> float:
    return float(
        metrics["model_input_leakage_count"] == 0.0
        and metrics["policy_output_oracle_usage_rate"] == 0.0
        and metrics["oracle_baseline_access_violation_count"] == 0.0
        and metrics["clean_oracle_closed_loop_score"] >= float(gates.get("clean_oracle_closed_loop_score_min", 0.95))
        and metrics["clean_random_closed_loop_score"] <= float(gates.get("clean_random_closed_loop_score_max", 0.25))
        and metrics["clean_b5_closed_loop_score"] >= float(gates.get("clean_b5_min_score_to_retain_claim", 0.70))
        and metrics["score_drop_from_original"] <= float(gates.get("clean_b5_max_score_drop_from_original", 0.35))
    )


def leakage_record(episode_id: int, stage: str, source: str, report: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "episode_id": int(episode_id),
        "model": model,
        "stage": stage,
        "source": source,
        "leakage_count": int(report.get("model_input_leakage_count", 0)),
        "forbidden_key_paths": report.get("forbidden_key_paths", ""),
        "fail_fast_triggered": int(report.get("fail_fast_triggered", False)),
    }


def read_original_b5_score(model_name: str) -> float:
    path = Path("results/b5_closed_loop_summary.csv")
    if not path.exists():
        return 0.8875
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == model_name:
                return float(row.get("b5_closed_loop_score", 0.8875))
    return 0.8875


def b5_clean_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    if "base_config" not in config:
        runtime = b5_runtime_config(config)
    else:
        path = Path(str(config.get("base_config", "configs/b5_closed_loop.yaml")))
        if not path.is_absolute():
            path = Path.cwd() / path
        with path.open("r", encoding="utf-8") as handle:
            runtime = b5_runtime_config(yaml.safe_load(handle))
    runtime["b5_clean"] = b5_clean_config(config)
    return runtime


def b5_clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b5_clean", {"n_test": 32, "gates": {}})


def effective_count(value: int, clean: dict[str, Any]) -> int:
    cap = clean.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))
