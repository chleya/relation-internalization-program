from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


POLICIES = [
    "b64_transfer_policy",
    "state_only",
    "mask_only",
    "trace_only",
    "random",
    "always_abstain",
    "conservative_uncertainty",
    "oracle",
]


def build_b6_4_adversarial_review(
    summary_path: str = "results/b6_4_transfer_summary.csv",
    records_path: str = "results/b6_4_transfer_records.csv",
    metrics_path: str = "results/b6_4_transfer_metrics.json",
    result_review_path: str = "results/b6_4_result_review.json",
) -> dict[str, Any]:
    summary = read_csv(Path(summary_path))
    records = read_csv(Path(records_path))
    metrics = read_json(Path(metrics_path))
    result_review = read_json(Path(result_review_path))
    conditions = sorted({row["condition"] for row in summary})
    remap_review = [review_condition(condition, summary) for condition in conditions]
    shortcut_equivalent = [row["condition"] for row in remap_review if row["shortcut_explainable"] and row["condition"] != "clean_reference"]
    stronger_transfer = [
        row["condition"]
        for row in remap_review
        if not row["shortcut_explainable"] and row["condition"] != "clean_reference" and row["oracle_gap"] <= 0.05
    ]
    review = {
        "decision": "harness_pass_not_strong_transfer_evidence",
        "summary": {
            "b64_policy_scores_all_remaps": metrics.get("b64_policy_mean_transfer_score", 0.0) == 1.0,
            "mean_baseline_transfer_gap": metrics.get("baseline_transfer_gap", 0.0),
            "mean_oracle_gap": metrics.get("oracle_gap", 0.0),
            "shortcut_equivalent_remap_count": len(shortcut_equivalent),
            "shortcut_equivalent_remaps": shortcut_equivalent,
            "stronger_transfer_remaps": stronger_transfer,
        },
        "remap_review": remap_review,
        "shortcut_audit": shortcut_audit(remap_review),
        "stronger_remap_audit": stronger_remap_audit(remap_review),
        "integrity": integrity_review(summary, records, result_review),
        "second_pass_needed": True,
        "recommended_second_pass_focus": [
            "visual_remap_hard",
            "risk_cue_remap_hard",
            "dynamics_remap_hard",
            "mask_visibility_remap_hard",
            "combined_remap_hard",
        ],
    }
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results/b6_4_adversarial_review.json").write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_4_ADVERSARIAL_RESULT_REVIEW.md").write_text(build_adversarial_report(review), encoding="utf-8")
    Path("reports/B6_4_SECOND_PASS_HARDENING_PLAN.md").write_text(build_second_pass_plan(), encoding="utf-8")
    return review


def review_condition(condition: str, summary: list[dict[str, str]]) -> dict[str, Any]:
    rows = {row["policy_name"]: row for row in summary if row["condition"] == condition}
    scores = {policy: as_float(rows.get(policy, {}).get("transfer_score")) for policy in POLICIES}
    b64 = scores["b64_transfer_policy"]
    shortcut_scores = {"state_only": scores["state_only"], "mask_only": scores["mask_only"], "trace_only": scores["trace_only"]}
    shortcut_source, shortcut_score = max(shortcut_scores.items(), key=lambda item: item[1])
    shortcut_explainable = b64 <= shortcut_score + 1e-9
    baseline_gap = as_float(rows.get("b64_transfer_policy", {}).get("baseline_transfer_gap"))
    oracle_gap = as_float(rows.get("b64_transfer_policy", {}).get("oracle_gap"))
    return {
        "condition": condition,
        "scores": scores,
        "transfer_drop": as_float(rows.get("b64_transfer_policy", {}).get("transfer_drop")),
        "baseline_transfer_gap": baseline_gap,
        "oracle_gap": oracle_gap,
        "remap_generalization_score": as_float(rows.get("b64_transfer_policy", {}).get("remap_generalization_score")),
        "anti_overfit_score": as_float(rows.get("b64_transfer_policy", {}).get("anti_overfit_score")),
        "shortcut_explainable": shortcut_explainable,
        "likely_shortcut_source": shortcut_source if shortcut_explainable else "none",
        "needed_hardening": needed_hardening(condition, shortcut_explainable, shortcut_source, baseline_gap),
    }


def needed_hardening(condition: str, shortcut_explainable: bool, shortcut_source: str, baseline_gap: float) -> str:
    if not shortcut_explainable and baseline_gap >= 0.20:
        return "retain_as_transfer_diagnostic"
    if condition == "visual_remap":
        return "remove trace-equivalent visual/state target cue and add appearance distractors"
    if condition == "risk_cue_remap":
        return "hide direct risk estimate and require feedback/history risk inference"
    if condition == "dynamics_remap":
        return "alter dynamics enough that trace-only and conservative rules fail"
    if condition == "mask_visibility_remap":
        return "hide answer-like mask fields plus state/trace substitutes"
    if condition == "combined_remap":
        return "increase combined remap pressure while preserving oracle interpretability"
    return f"reduce {shortcut_source} shortcut"


def shortcut_audit(remap_review: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {row["condition"]: row for row in remap_review}
    return {
        "visual_shortcut_source": lookup.get("visual_remap", {}).get("likely_shortcut_source", "missing"),
        "risk_cue_shortcut_source": lookup.get("risk_cue_remap", {}).get("likely_shortcut_source", "missing"),
        "dynamics_shortcut_source": lookup.get("dynamics_remap", {}).get("likely_shortcut_source", "missing"),
        "mask_visibility_shortcut_source": lookup.get("mask_visibility_remap", {}).get("likely_shortcut_source", "missing"),
        "shortcut_equivalent_remap_count": sum(1 for row in remap_review if row["shortcut_explainable"] and row["condition"] != "clean_reference"),
    }


def stronger_remap_audit(remap_review: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {row["condition"]: row for row in remap_review}
    return {
        "delay_profile_remap": explain_stronger_remap(lookup.get("delay_profile_remap")),
        "indirect_path_remap": explain_stronger_remap(lookup.get("indirect_path_remap")),
        "combined_remap": explain_stronger_remap(lookup.get("combined_remap")),
    }


def explain_stronger_remap(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"present": False}
    return {
        "present": True,
        "baseline_transfer_gap": row["baseline_transfer_gap"],
        "oracle_gap": row["oracle_gap"],
        "mechanism_readout": "delayed_credit_or_hidden_indirect_history" if not row["shortcut_explainable"] else "shortcut_matched",
        "interpretation": "stronger harness signal, still synthetic toy transfer",
    }


def integrity_review(summary: list[dict[str, str]], records: list[dict[str, str]], result_review: dict[str, Any]) -> dict[str, Any]:
    return {
        "forbidden_reference_count_max": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
        "poisoned_evaluator_invariance_all_pass": all(str(row["poisoned_evaluator_invariance_pass"]).lower() in {"true", "1"} for row in summary),
        "no_sample_metric_count_total": sum(int(row["no_sample_metric_count"]) for row in summary),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "remap_leakage_count_total": sum(int(row["remap_leakage_count"]) for row in summary),
        "selected_source_counts": dict(Counter(row.get("transfer_source", "none") for row in records if row["policy_name"] == "b64_transfer_policy")),
        "review_claim_boundary": result_review.get("claim_boundary"),
    }


def build_adversarial_report(review: dict[str, Any]) -> str:
    lines = [
        "# B6.4 Adversarial Result Review",
        "",
        "## Decision",
        "- B6.4 first pass is a transfer harness pass, not strong transfer evidence.",
        f"- second_pass_needed: {str(review['second_pass_needed']).lower()}",
        "",
        "## Summary",
    ]
    for key, value in review["summary"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Remap-by-Remap Audit"])
    for row in review["remap_review"]:
        lines.append(
            f"- {row['condition']}: b64={row['scores']['b64_transfer_policy']:.3f}, "
            f"state={row['scores']['state_only']:.3f}, mask={row['scores']['mask_only']:.3f}, "
            f"trace={row['scores']['trace_only']:.3f}, oracle={row['scores']['oracle']:.3f}, "
            f"shortcut_explainable={str(row['shortcut_explainable']).lower()}, "
            f"likely_shortcut_source={row['likely_shortcut_source']}, needed_hardening={row['needed_hardening']}"
        )
    lines.extend(["", "## Shortcut Audit"])
    for key, value in review["shortcut_audit"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Stronger Remap Audit"])
    for key, value in review["stronger_remap_audit"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Integrity"])
    for key, value in review["integrity"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(
        [
            "",
            "## Caveats",
            "- B6.4 scores 1.000 on all remaps, which is too clean for strong transfer evidence.",
            "- mean_baseline_transfer_gap is modest, so shortcut baselines still explain part of the result.",
            "- visual_remap, risk_cue_remap, dynamics_remap, and mask_visibility_remap remain shortcut-explainable.",
            "- delay_profile_remap, indirect_path_remap, and combined_remap show stronger baseline separation, but remain synthetic toy diagnostics.",
            "- No real-world generalization, robotics, safety certification, construction-site autonomy, or deployment claim is supported.",
            "",
        ]
    )
    return "\n".join(lines)


def build_second_pass_plan() -> str:
    return """# B6.4 Second-Pass Hardening Plan

## Purpose

B6.4 first pass is a transfer harness pass, not strong transfer evidence. The second pass should harden shortcut-equivalent remaps while keeping the task toy-to-toy.

## 1. visual_remap_hard

- remove direct public state target cue
- alter spatial encoding
- introduce appearance distractors
- test state_only drop

## 2. risk_cue_remap_hard

- invert or rotate risk cue encoding
- hide direct risk estimate
- require feedback/history risk inference
- test risk cue transfer

## 3. dynamics_remap_hard

- alter dynamics enough that short-horizon, trace-only, and conservative rules fail
- preserve latent operational structure
- test trace/history contribution

## 4. mask_visibility_remap_hard

- remove answer-like mask fields
- hide unsafe/irreversible/cost/indirect target
- test fallback inference and history

## 5. combined_remap_hard

- combine visual, risk, dynamics, mask, and delay remaps
- allow expected score drop
- measure relative baseline gap

## 6. Anti-Overfit Criteria

- transfer evidence requires b64_transfer_policy to beat state_only, mask_only, and trace_only by a meaningful margin
- shortcut baselines must degrade under the remap they are supposed to fail
- oracle_gap must remain interpretable
- high b64 score alone is insufficient
- no real-world risk intelligence, robotics, safety certification, construction-site autonomy, or deployable control claim is allowed
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
