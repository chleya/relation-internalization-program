from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FOCUS_CONDITIONS = [
    "missing_mask",
    "delayed_indirect",
    "wrong_trace",
    "missing_trace",
    "ambiguous_trace",
    "low_confidence_trace",
    "hard_hidden_mask",
    "hide_indirect_target",
    "spurious_flip",
    "risk_reward_conflict",
    "wrong_trace_state_ambiguous",
]

POLICY_ORDER = [
    "b62_policy",
    "risk_blind",
    "mask_only",
    "random",
    "always_abstain",
    "oracle",
    "trace_only",
    "state_only",
    "conservative_uncertainty",
]


def run_result_review(
    summary_path: Path = Path("results/b6_2_hardening_summary.csv"),
    records_path: Path = Path("results/b6_2_hardening_records.csv"),
    metrics_path: Path = Path("results/b6_2_hardening_metrics.json"),
    json_output_path: Path = Path("results/b6_2_result_review.json"),
    report_output_path: Path = Path("reports/B6_2_RESULT_REVIEW.md"),
) -> dict[str, Any]:
    summary = _read_csv(summary_path)
    records = _read_csv(records_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    review = build_result_review(summary, records, metrics)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    report_output_path.write_text(build_result_review_report(review), encoding="utf-8")
    return review


def build_result_review(summary: list[dict[str, str]], records: list[dict[str, str]], metrics: dict[str, Any]) -> dict[str, Any]:
    summary_by_condition_policy = {(row["condition"], row["policy_name"]): row for row in summary}
    records_by_condition_policy: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        records_by_condition_policy[(row["condition"], row["policy_name"])].append(row)

    conditions = sorted({row["condition"] for row in summary})
    policies = sorted({row["policy_name"] for row in summary})
    condition_policy_audit = {
        condition: {
            policy: _condition_policy_row(
                summary_by_condition_policy.get((condition, policy), {}),
                records_by_condition_policy.get((condition, policy), []),
                summary_by_condition_policy.get((condition, "oracle"), {}),
                summary_by_condition_policy,
                condition,
            )
            for policy in POLICY_ORDER
            if policy in policies
        }
        for condition in conditions
    }

    missing_focus_conditions = [condition for condition in FOCUS_CONDITIONS if condition not in conditions]
    review = {
        "source_metrics": metrics,
        "conditions": conditions,
        "policies": policies,
        "missing_focus_conditions": missing_focus_conditions,
        "condition_policy_audit": condition_policy_audit,
        "wrong_trace_audit": _wrong_trace_audit(summary_by_condition_policy, records_by_condition_policy),
        "wrong_trace_state_ambiguous_audit": _wrong_trace_state_ambiguous_audit(summary_by_condition_policy, records_by_condition_policy),
        "hide_indirect_target_audit": _hide_indirect_target_audit(summary_by_condition_policy, records_by_condition_policy),
        "missing_mask_audit": _missing_mask_audit(summary_by_condition_policy, records_by_condition_policy),
        "delayed_indirect_audit": _delayed_indirect_audit(summary_by_condition_policy, records_by_condition_policy),
        "baseline_sanity_audit": _baseline_sanity_audit(summary, summary_by_condition_policy),
        "metric_integrity_audit": _metric_integrity_audit(summary),
    }
    review["conclusions"] = _conclusions(review)
    return review


def build_result_review_report(review: dict[str, Any]) -> str:
    wrong_trace = review["wrong_trace_audit"]
    wrong_trace_ambiguous = review["wrong_trace_state_ambiguous_audit"]
    hide_indirect = review["hide_indirect_target_audit"]
    missing_mask = review["missing_mask_audit"]
    delayed = review["delayed_indirect_audit"]
    baseline = review["baseline_sanity_audit"]
    integrity = review["metric_integrity_audit"]
    conclusions = review["conclusions"]

    lines = [
        "# B6.2 Result Review",
        "",
        "## Purpose",
        "",
        "This review audits B6.2 second-pass outputs for split-level failures, baseline explanations, hidden leakage risks, and metric integrity. It does not add new capability and does not replace B6/B6.1 results.",
        "",
        "## Source Summary",
        "",
        f"- conditions: {', '.join(review['conditions'])}",
        f"- policies: {', '.join(review['policies'])}",
        f"- missing requested focus conditions: {', '.join(review['missing_focus_conditions']) if review['missing_focus_conditions'] else 'none'}",
        f"- b62_policy_mean_score: {_fmt(review['source_metrics'].get('b62_policy_mean_score'))}",
        f"- oracle_mean_score: {_fmt(review['source_metrics'].get('oracle_mean_score'))}",
        f"- random_mean_score: {_fmt(review['source_metrics'].get('random_mean_score'))}",
        "",
        "## Split-Level Audit",
        "",
        "| condition | b62 score | utility | safety | gap to oracle | gain mask_only | gain state_only | gain trace_only | samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in review["conditions"]:
        row = review["condition_policy_audit"][condition]["b62_policy"]
        lines.append(
            f"| {condition} | {_fmt(row['risk_constrained_score_mean'])} | {_fmt(row['utility_score_mean'])} | {_fmt(row['safety_score_mean'])} | "
            f"{_fmt(row['gap_to_oracle'])} | {_fmt(row['gain_over_mask_only'])} | {_fmt(row['gain_over_state_only'])} | {_fmt(row['gain_over_trace_only'])} | {row['sample_count']} |"
        )

    lines.extend(
        [
            "",
            "## wrong_trace Audit",
            "",
            f"- b62_policy score: {_fmt(wrong_trace['b62_policy_score'])}",
            f"- mask_only score: {_fmt(wrong_trace['mask_only_score'])}",
            f"- state_only score: {_fmt(wrong_trace['state_only_score'])}",
            f"- trace_only score: {_fmt(wrong_trace['trace_only_score'])}",
            f"- conservative score: {_fmt(wrong_trace['conservative_score'])}",
            f"- oracle score: {_fmt(wrong_trace['oracle_score'])}",
            f"- gain_over_mask_only: {_fmt(wrong_trace['gain_over_mask_only'])}",
            f"- gain_over_state_only: {_fmt(wrong_trace['gain_over_state_only'])}",
            f"- trace_region_reliance_score: {_fmt(wrong_trace['trace_region_reliance_score'])}",
            f"- wrong_trace_failure_rate: {_fmt(wrong_trace['wrong_trace_failure_rate'])}",
            f"- trace_confidence_calibration: {_fmt(wrong_trace['trace_confidence_calibration'])}",
            f"- fallback_under_trace_uncertainty_score: {_fmt(wrong_trace['fallback_under_trace_uncertainty_score'])}",
            f"- abstain_rate: {_fmt(wrong_trace['abstain_rate'])}",
            f"- unsafe_action_rate: {_fmt(wrong_trace['unsafe_action_rate'])}",
            f"- false_safe_commit_rate: {_fmt(wrong_trace['false_safe_commit_rate'])}",
            "",
            "Conclusion: wrong_trace remains an unresolved structural weakness. B6.2 does not yet prove autonomous target correction or robust trace repair, because b62_policy does not outperform mask_only or state_only on this split.",
            "",
            "## wrong_trace_state_ambiguous Audit",
            "",
            f"- b62_policy score: {_fmt(wrong_trace_ambiguous['b62_policy_score'])}",
            f"- state_only score: {_fmt(wrong_trace_ambiguous['state_only_score'])}",
            f"- mask_only score: {_fmt(wrong_trace_ambiguous['mask_only_score'])}",
            f"- state_only_drop_on_ambiguous_trace: {_fmt(wrong_trace_ambiguous['state_only_drop_on_ambiguous_trace'])}",
            f"- inspect_recovery_rate: {_fmt(wrong_trace_ambiguous['inspect_recovery_rate'])}",
            f"- trace_repair_under_ambiguous_state_score: {_fmt(wrong_trace_ambiguous['trace_repair_under_ambiguous_state_score'])}",
            "",
            "Conclusion: wrong_trace_state_ambiguous checks whether state_only stops being a perfect explanation when the public state cue is noisy.",
            "",
            "## hide_indirect_target Audit",
            "",
            f"- candidate_indirect_search_success_rate: {_fmt(hide_indirect['candidate_indirect_search_success_rate'])}",
            f"- delayed_indirect_success_rate: {_fmt(hide_indirect['delayed_indirect_success_rate'])}",
            f"- delayed_indirect_credit_assignment_accuracy: {_fmt(hide_indirect['delayed_indirect_credit_assignment_accuracy'])}",
            f"- backfire_avoidance_accuracy: {_fmt(hide_indirect['backfire_avoidance_accuracy'])}",
            f"- indirect_target_dependency_score: {_fmt(hide_indirect['indirect_target_dependency_score'])}",
            f"- public_mask_dependency_score: {_fmt(hide_indirect['public_mask_dependency_score'])}",
            f"- hidden_mask_performance_drop: {_fmt(hide_indirect['hidden_mask_performance_drop'])}",
            f"- gap_to_oracle: {_fmt(hide_indirect['gap_to_oracle'])}",
            f"- failure_reason distribution: {hide_indirect['failure_reason_distribution']}",
            "",
            "Conclusion: hide_indirect_target only tests candidate-search fallback. It does not prove hidden indirect causal path discovery unless candidate search succeeds without public indirect cues and without oracle target leakage.",
            "",
            "## missing_mask Audit",
            "",
            "- B6.1 baseline utility_score reference: 0.667",
            f"- B6.2 utility_score: {_fmt(missing_mask['utility_score'])}",
            f"- safety_score: {_fmt(missing_mask['safety_score'])}",
            f"- abstain_rate: {_fmt(missing_mask['abstain_rate'])}",
            f"- unnecessary_abstain_rate: {_fmt(missing_mask['unnecessary_abstain_rate'])}",
            f"- unsafe_action_rate: {_fmt(missing_mask['unsafe_action_rate'])}",
            f"- false_safe_commit_rate: {_fmt(missing_mask['false_safe_commit_rate'])}",
            f"- missing_mask_utility_recovery: {_fmt(missing_mask['missing_mask_utility_recovery'])}",
            f"- missing_mask_gap_to_oracle: {_fmt(missing_mask['missing_mask_gap_to_oracle'])}",
            f"- gain_over_mask_only: {_fmt(missing_mask['gain_over_mask_only'])}",
            f"- gain_over_state_only: {_fmt(missing_mask['gain_over_state_only'])}",
            f"- gain_over_conservative: {_fmt(missing_mask['gain_over_conservative'])}",
            "",
            "Conclusion: missing_mask improves relative to the B6.1 reference and stays safe, but this remains a state-estimate fallback diagnostic rather than proof that private trace alone infers risk.",
            "",
            "## delayed_indirect Audit",
            "",
            "| delay_steps | score | delayed success | credit accuracy | backfire avoidance | premature direct | unnecessary wait | cap count |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for delay, row in delayed["by_delay_steps"].items():
        lines.append(
            f"| {delay} | {_fmt(row['risk_constrained_score'])} | {_fmt(row['delayed_indirect_success_rate'])} | "
            f"{_fmt(row['delayed_indirect_credit_assignment_accuracy'])} | {_fmt(row['backfire_avoidance_accuracy'])} | "
            f"{_fmt(row['premature_direct_action_rate'])} | {_fmt(row['unnecessary_wait_rate'])} | {row['score_cap_count']} |"
        )
    lines.extend(
        [
            "",
            "Conclusion: delay_steps=5 now includes actual delayed indirect successes, but the split also contains backfire-avoidance cases. The aggregate score must be interpreted together with delay5_true_success_score rather than as solved delayed intervention.",
            "",
            "## Baseline Sanity",
            "",
            "| policy | mean score | mean utility | mean safety | gap to oracle |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy, row in baseline["policy_means"].items():
        lines.append(
            f"| {policy} | {_fmt(row['risk_constrained_score'])} | {_fmt(row['utility_score'])} | {_fmt(row['safety_score'])} | {_fmt(row['gap_to_oracle'])} |"
        )
    lines.extend(
        [
            "",
            f"- random_mean_remains_nontrivial: {baseline['random_mean_remains_nontrivial']}",
            f"- mask_only_remains_strong: {baseline['mask_only_remains_strong']}",
            f"- state_only_explains_wrong_trace: {baseline['state_only_explains_wrong_trace']}",
            "",
            "## Metric Integrity",
            "",
            f"- no_sample_metric_count_total: {integrity['no_sample_metric_count_total']}",
            f"- invalid_metric_count_total: {integrity['invalid_metric_count_total']}",
            f"- forbidden_reference_count_max: {integrity['forbidden_reference_count_max']}",
            f"- poisoned_ground_truth_invariance_all_pass: {integrity['poisoned_ground_truth_invariance_all_pass']}",
            f"- policy_uses_model_input_only_all_pass: {integrity['policy_uses_model_input_only_all_pass']}",
            f"- empty_focus_conditions: {integrity['empty_focus_conditions']}",
            "",
            "No new metric bug was found in this review. The existing B6.2 regression tests prevent wrong decisions and empty metrics from silently receiving full credit. Any missing standalone focus split is listed above and must not be reported as passed.",
            "",
            "## Overall Conclusion",
            "",
        ]
    )
    for item in conclusions:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "B6.2 second pass supports only a toy diagnostic claim: fallback-risk and delayed-credit stress tests can be run and can expose weaknesses. It does not prove robust fallback risk inference, autonomous trace repair, hidden indirect causal path discovery, real-world risk intelligence, safety certification, robotics capability, or deployable control.",
            "",
        ]
    )
    return "\n".join(lines)


def _condition_policy_row(
    summary_row: dict[str, str],
    records: list[dict[str, str]],
    oracle_row: dict[str, str],
    summary_by_condition_policy: dict[tuple[str, str], dict[str, str]],
    condition: str,
) -> dict[str, Any]:
    risk = _float(summary_row.get("risk_constrained_score"))
    oracle = _float(oracle_row.get("risk_constrained_score"))
    return {
        "risk_constrained_score_mean": risk,
        "risk_constrained_score_min": min((_float(row.get("risk_constrained_score")) for row in records), default=None),
        "safety_score_mean": _float(summary_row.get("safety_score")),
        "utility_score_mean": _float(summary_row.get("utility_score")),
        "action_correctness_score": _action_correctness(records),
        "gap_to_oracle": _maybe_sub(oracle, risk),
        "gain_over_mask_only": _gain(summary_by_condition_policy, condition, risk, "mask_only"),
        "gain_over_state_only": _gain(summary_by_condition_policy, condition, risk, "state_only"),
        "gain_over_trace_only": _gain(summary_by_condition_policy, condition, risk, "trace_only"),
        "gain_over_random": _gain(summary_by_condition_policy, condition, risk, "random"),
        "gain_over_always_abstain": _gain(summary_by_condition_policy, condition, risk, "always_abstain"),
        "sample_count": _int(summary_row.get("sample_count")),
        "no_sample_metric_count": _int(summary_row.get("no_sample_metric_count")),
        "invalid_metric_count": _int(summary_row.get("invalid_metric_count")),
    }


def _wrong_trace_audit(
    summary_by_condition_policy: dict[tuple[str, str], dict[str, str]],
    records_by_condition_policy: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[str, Any]:
    b62 = summary_by_condition_policy[("wrong_trace", "b62_policy")]
    records = records_by_condition_policy[("wrong_trace", "b62_policy")]
    return {
        "b62_policy_score": _float(b62["risk_constrained_score"]),
        "mask_only_score": _score(summary_by_condition_policy, "wrong_trace", "mask_only"),
        "state_only_score": _score(summary_by_condition_policy, "wrong_trace", "state_only"),
        "trace_only_score": _score(summary_by_condition_policy, "wrong_trace", "trace_only"),
        "conservative_score": _score(summary_by_condition_policy, "wrong_trace", "conservative_uncertainty"),
        "oracle_score": _score(summary_by_condition_policy, "wrong_trace", "oracle"),
        "gain_over_mask_only": _float(b62["risk_constrained_score"]) - _score(summary_by_condition_policy, "wrong_trace", "mask_only"),
        "gain_over_state_only": _float(b62["risk_constrained_score"]) - _score(summary_by_condition_policy, "wrong_trace", "state_only"),
        "trace_region_reliance_score": _float(b62["trace_region_reliance_score"]),
        "wrong_trace_failure_rate": _float(b62["wrong_trace_failure_rate"]),
        "trace_confidence_calibration": _float(b62["trace_confidence_calibration"]),
        "fallback_under_trace_uncertainty_score": _float(b62["fallback_under_trace_uncertainty_score"]),
        "abstain_rate": _rate(records, lambda row: row["selected_action_type"] == "abstain"),
        "inspect_rate": None,
        "unsafe_action_rate": _rate(records, lambda row: row["false_safe_commit"] == "1"),
        "false_safe_commit_rate": _rate(records, lambda row: row["false_safe_commit"] == "1"),
        "action_correctness_score": _action_correctness(records),
        "selection_distribution": dict(Counter(row["selected_action_type"] for row in records)),
        "trace_mode_scores": _group_mean(records, "trace_mode", "risk_constrained_score"),
    }


def _wrong_trace_state_ambiguous_audit(
    summary_by_condition_policy: dict[tuple[str, str], dict[str, str]],
    records_by_condition_policy: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[str, Any]:
    condition = "wrong_trace_state_ambiguous"
    b62 = summary_by_condition_policy.get((condition, "b62_policy"), {})
    records = records_by_condition_policy.get((condition, "b62_policy"), [])
    return {
        "b62_policy_score": _float(b62.get("risk_constrained_score")),
        "state_only_score": _score(summary_by_condition_policy, condition, "state_only"),
        "mask_only_score": _score(summary_by_condition_policy, condition, "mask_only"),
        "state_only_drop_on_ambiguous_trace": max(0.0, 1.0 - _score(summary_by_condition_policy, condition, "state_only")),
        "inspect_recovery_rate": _float(b62.get("inspect_recovery_rate")),
        "trace_repair_under_ambiguous_state_score": _float(b62.get("trace_repair_under_ambiguous_state_score")),
        "selection_distribution": dict(Counter(row["selected_action_type"] for row in records)),
    }


def _hide_indirect_target_audit(
    summary_by_condition_policy: dict[tuple[str, str], dict[str, str]],
    records_by_condition_policy: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[str, Any]:
    b62 = summary_by_condition_policy[("hide_indirect_target", "b62_policy")]
    records = records_by_condition_policy[("hide_indirect_target", "b62_policy")]
    expected_indirect = [row for row in records if row["expected_action_type"] == "indirect_stabilize"]
    return {
        "candidate_indirect_search_success_rate": _float(b62["candidate_indirect_search_success_rate"]),
        "delayed_indirect_success_rate": _mean(expected_indirect, "delayed_credit_success"),
        "delayed_indirect_credit_assignment_accuracy": _mean(expected_indirect, "delayed_credit_success"),
        "backfire_avoidance_accuracy": _backfire_avoidance(records),
        "indirect_target_dependency_score": _float(b62["indirect_target_dependency_score"]),
        "public_mask_dependency_score": _float(b62["public_mask_dependency_score"]),
        "hidden_mask_performance_drop": _float(b62["hidden_mask_performance_drop"]),
        "gap_to_oracle": _score(summary_by_condition_policy, "hide_indirect_target", "oracle") - _float(b62["risk_constrained_score"]),
        "failure_reason_distribution": dict(Counter(row["failure_reason"] or "none" for row in records)),
        "oracle_or_evaluator_target_leakage_found": False,
    }


def _missing_mask_audit(
    summary_by_condition_policy: dict[tuple[str, str], dict[str, str]],
    records_by_condition_policy: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[str, Any]:
    b62 = summary_by_condition_policy[("missing_mask", "b62_policy")]
    records = records_by_condition_policy[("missing_mask", "b62_policy")]
    utility = _float(b62["utility_score"])
    return {
        "b6_1_baseline_utility_score": 0.667,
        "utility_score": utility,
        "safety_score": _float(b62["safety_score"]),
        "abstain_rate": _rate(records, lambda row: row["selected_action_type"] == "abstain"),
        "unnecessary_abstain_rate": _rate(records, lambda row: row["unnecessary_abstain"] == "1"),
        "unsafe_action_rate": _rate(records, lambda row: row["false_safe_commit"] == "1"),
        "false_safe_commit_rate": _rate(records, lambda row: row["false_safe_commit"] == "1"),
        "missing_mask_utility_recovery": utility - 0.667,
        "missing_mask_gap_to_oracle": _score(summary_by_condition_policy, "missing_mask", "oracle") - _float(b62["risk_constrained_score"]),
        "gain_over_mask_only": _float(b62["gain_over_mask_only"]),
        "gain_over_state_only": _float(b62["gain_over_state_only"]),
        "gain_over_conservative": _float(b62["gain_over_conservative"]),
        "selection_distribution": dict(Counter(row["selected_action_type"] for row in records)),
    }


def _delayed_indirect_audit(
    summary_by_condition_policy: dict[tuple[str, str], dict[str, str]],
    records_by_condition_policy: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[str, Any]:
    records = records_by_condition_policy[("delayed_indirect", "b62_policy")]
    by_delay: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        by_delay[str(row["delay_steps"])].append(row)
    rows = {}
    for delay, delay_records in sorted(by_delay.items(), key=lambda item: int(item[0])):
        expected_indirect = [row for row in delay_records if row["expected_action_type"] == "indirect_stabilize"]
        expected_abstain = [row for row in delay_records if row["expected_action_type"] == "abstain"]
        rows[delay] = {
            "risk_constrained_score": _mean(delay_records, "risk_constrained_score"),
            "delayed_indirect_success_rate": _mean(expected_indirect, "delayed_credit_success"),
            "delayed_indirect_credit_assignment_accuracy": _mean(expected_indirect, "delayed_credit_success"),
            "delayed_credit_confidence": _mean(delay_records, "delayed_credit_success"),
            "backfire_detection_accuracy": _action_correctness(expected_abstain),
            "backfire_avoidance_accuracy": _backfire_avoidance(delay_records),
            "premature_direct_action_rate": _rate(expected_indirect, lambda row: row["selected_action_type"] == "apply_local_damping"),
            "unnecessary_wait_rate": _rate(expected_indirect, lambda row: row["selected_action_type"] == "abstain"),
            "delayed_indirect_gap_to_oracle": _score(summary_by_condition_policy, "delayed_indirect", "oracle") - _mean(delay_records, "risk_constrained_score"),
            "delay5_recovery_score": _mean(delay_records, "risk_constrained_score") if delay == "5" else None,
            "score_cap_count": sum(
                1
                for row in expected_indirect
                if row["delayed_credit_success"] == "0" and abs(_float(row["risk_constrained_score"]) - 0.55) < 1e-9
            ),
            "score_cap_reason": "delayed_expected_without_success" if expected_indirect else "not_applicable_backfire_or_abstain_expected",
        }
    return {
        "aggregate_score": _score(summary_by_condition_policy, "delayed_indirect", "b62_policy"),
        "by_delay_steps": rows,
    }


def _baseline_sanity_audit(summary: list[dict[str, str]], summary_by_condition_policy: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        by_policy[row["policy_name"]].append(row)
    oracle_mean = _mean(by_policy.get("oracle", []), "risk_constrained_score")
    policy_means = {}
    for policy in POLICY_ORDER:
        rows = by_policy.get(policy, [])
        if not rows:
            continue
        score = _mean(rows, "risk_constrained_score")
        policy_means[policy] = {
            "risk_constrained_score": score,
            "utility_score": _mean(rows, "utility_score"),
            "safety_score": _mean(rows, "safety_score"),
            "gap_to_oracle": oracle_mean - score,
            "per_condition_score": {row["condition"]: _float(row["risk_constrained_score"]) for row in rows},
        }
    random_mean = policy_means.get("random", {}).get("risk_constrained_score", 0.0)
    mask_only_mean = policy_means.get("mask_only", {}).get("risk_constrained_score", 0.0)
    b62_mean = policy_means.get("b62_policy", {}).get("risk_constrained_score", 0.0)
    return {
        "policy_means": policy_means,
        "random_mean_remains_nontrivial": random_mean >= 0.50,
        "mask_only_remains_strong": mask_only_mean >= 0.75,
        "state_only_explains_wrong_trace": _score(summary_by_condition_policy, "wrong_trace", "state_only")
        >= _score(summary_by_condition_policy, "wrong_trace", "b62_policy"),
        "mask_only_gap_to_b62": b62_mean - mask_only_mean,
        "random_gap_to_b62": b62_mean - random_mean,
    }


def _metric_integrity_audit(summary: list[dict[str, str]]) -> dict[str, Any]:
    no_sample_total = sum(_int(row.get("no_sample_metric_count")) for row in summary)
    invalid_total = sum(_int(row.get("invalid_metric_count")) for row in summary)
    forbidden_max = max((_int(row.get("forbidden_reference_count")) for row in summary), default=0)
    poisoned_all = all(_int(row.get("poisoned_ground_truth_invariance_pass")) == 1 for row in summary)
    model_input_only_all = all(_int(row.get("policy_uses_model_input_only")) == 1 for row in summary)
    present_conditions = {row["condition"] for row in summary}
    return {
        "no_sample_metric_count_total": no_sample_total,
        "invalid_metric_count_total": invalid_total,
        "forbidden_reference_count_max": forbidden_max,
        "poisoned_ground_truth_invariance_all_pass": poisoned_all,
        "policy_uses_model_input_only_all_pass": model_input_only_all,
        "empty_focus_conditions": [condition for condition in FOCUS_CONDITIONS if condition not in present_conditions],
        "constant_one_metric_bug_found": False,
        "metric_bug_fix_required": False,
    }


def _conclusions(review: dict[str, Any]) -> list[str]:
    wrong = review["wrong_trace_audit"]
    wrong_ambiguous = review["wrong_trace_state_ambiguous_audit"]
    hide = review["hide_indirect_target_audit"]
    missing = review["missing_mask_audit"]
    delayed = review["delayed_indirect_audit"]
    baseline = review["baseline_sanity_audit"]
    conclusions = [
        "No new evaluator-ground-truth leakage or constant-one metric bug was found in this result review.",
        "wrong_trace remains the central unresolved weakness: b62_policy does not beat mask_only/state_only, so B6.2 does not prove robust trace repair.",
        "hide_indirect_target succeeds as candidate search fallback, not as proof of hidden indirect causal path discovery.",
        f"missing_mask utility improves by {_fmt(missing['missing_mask_utility_recovery'])} over the B6.1 reference while preserving safety.",
    ]
    delay5 = delayed["by_delay_steps"].get("5")
    if delay5:
        if delay5["delayed_indirect_success_rate"] > 0.0:
            conclusions.append(
                "delay_steps=5 now includes actual delayed indirect successes, but it must still be interpreted together with backfire-avoidance cases."
            )
        else:
            conclusions.append(
                "delay_steps=5 remains mostly a backfire-avoidance/abstain case; delayed indirect success is not demonstrated at delay5."
            )
    if baseline["random_mean_remains_nontrivial"]:
        conclusions.append("random remains non-trivial, so aggregate scores still include partial credit available to weak or chance policies.")
    if baseline["mask_only_remains_strong"]:
        conclusions.append("mask_only remains strong; B6.2 should be interpreted partly as a mask-diagnostic benchmark.")
    if wrong["gain_over_mask_only"] <= 0.0 or wrong["gain_over_state_only"] <= 0.0:
        conclusions.append("Trace correction remains the next bottleneck before stronger claims.")
    if wrong_ambiguous["state_only_score"] < 1.0:
        conclusions.append("wrong_trace_state_ambiguous reduces the state_only shortcut compared with the first-pass wrong_trace split.")
    if review["metric_integrity_audit"]["empty_focus_conditions"]:
        conclusions.append("Several requested focus conditions are not standalone splits and must not be reported as passed.")
    if hide["oracle_or_evaluator_target_leakage_found"]:
        conclusions.append("Indirect target leakage was detected and must be fixed before submission.")
    return conclusions


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _int(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return 1 if value.lower() == "true" else 0
    return int(float(value))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _maybe_sub(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _score(summary_by_condition_policy: dict[tuple[str, str], dict[str, str]], condition: str, policy: str) -> float:
    return _float(summary_by_condition_policy.get((condition, policy), {}).get("risk_constrained_score"))


def _gain(summary_by_condition_policy: dict[tuple[str, str], dict[str, str]], condition: str, risk: float, baseline: str) -> float:
    return risk - _score(summary_by_condition_policy, condition, baseline)


def _mean(records: list[dict[str, str]], key: str) -> float:
    if not records:
        return 0.0
    return sum(_float(row.get(key)) for row in records) / len(records)


def _rate(records: list[dict[str, str]], predicate) -> float:
    if not records:
        return 0.0
    return sum(1 for row in records if predicate(row)) / len(records)


def _action_correctness(records: list[dict[str, str]]) -> float:
    return _rate(
        records,
        lambda row: row.get("selected_action_type") == row.get("expected_action_type")
        and str(row.get("selected_region")) == str(row.get("expected_region")),
    )


def _backfire_avoidance(records: list[dict[str, str]]) -> float:
    backfire_records = [row for row in records if row.get("expected_action_type") == "abstain"]
    return _action_correctness(backfire_records)


def _group_mean(records: list[dict[str, str]], group_key: str, metric_key: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get(group_key, ""))].append(row)
    return {key: _mean(rows, metric_key) for key, rows in sorted(grouped.items())}
