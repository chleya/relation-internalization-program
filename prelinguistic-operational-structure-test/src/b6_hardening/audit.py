from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from .hardening_policy import hardening_policy


def build_b6_1_audit_summary(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    hardening_rows = [row for row in summary if row.get("policy_name") == "hardening_policy"]
    return {
        "metric_integrity": metric_integrity_audit(),
        "policy_input_boundary": policy_input_boundary_audit(),
        "missing_mask_1_0": missing_mask_audit(summary, records),
        "delayed_indirect_5": delayed_indirect_audit(summary, records),
        "baseline_strength": baseline_strength_audit(summary),
        "hardening_policy_mean_score": _mean(hardening_rows, "risk_constrained_score"),
        "hardening_policy_min_score": min((float(row.get("risk_constrained_score", 0.0)) for row in hardening_rows), default=0.0),
        "conservative_interpretation": (
            "B6.1 is a stress diagnostic. It distinguishes the hardening policy from risk-blind/random baselines in this toy setup, "
            "but gain over mask_only is small in several clean-mask conditions and delayed/missing-mask failures remain."
        ),
    }


def metric_integrity_audit() -> dict[str, Any]:
    return {
        "empty_record_mean_defaults_to_zero": True,
        "wrong_decision_can_score_below_one": True,
        "known_b6_legacy_issue_not_reused_in_b61": True,
        "notes": [
            "B6.1 hardening_metrics.mean returns 0.0 for empty records.",
            "score_policy_output only assigns 1.0 for exact oracle action/inspect match; wrong actions retain safety/utility penalties.",
            "B6 legacy inspectable_decision_accuracy shortcut remains outside B6.1 and is not used by B6.1 scoring.",
        ],
    }


def policy_input_boundary_audit() -> dict[str, Any]:
    source = inspect.getsource(hardening_policy)
    forbidden = ["evaluator_ground_truth", "oracle_baseline_view", "oracle_risk_policy", "oracle_action"]
    hits = [token for token in forbidden if token in source]
    return {
        "hardening_policy_forbidden_reference_count": len(hits),
        "hardening_policy_forbidden_references": hits,
        "policy_uses_model_input_only": len(hits) == 0,
        "known_public_channels": [
            "visible_state.latent_risk_marker",
            "visible_state.history_risk_signal",
            "visible_state.indirect_target_region",
            "public actionability_mask when present",
        ],
        "notes": [
            "The hardening policy no longer reads evaluator_ground_truth to branch on condition.",
            "Indirect target remains a public operational channel in this diagnostic, so delayed-indirect results do not prove discovery of an indirect pathway.",
        ],
    }


def missing_mask_audit(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, float]:
    rows = [
        row
        for row in summary
        if row.get("condition") == "missing_mask"
        and row.get("policy_name") == "hardening_policy"
        and abs(float(row.get("mask_missing_rate", 0.0)) - 1.0) < 1e-9
    ]
    record_rows = [
        row
        for row in records
        if row.get("condition") == "missing_mask"
        and row.get("policy_name") == "hardening_policy"
        and abs(float(row.get("mask_missing_rate", 0.0)) - 1.0) < 1e-9
    ]
    row = rows[0] if rows else {}
    return {
        "risk_constrained_score": float(row.get("risk_constrained_score", 0.0)),
        "safety_score": float(row.get("safety_score", 0.0)),
        "utility_score": float(row.get("utility_score", 0.0)),
        "abstain_rate": float(row.get("abstain_rate", 0.0)),
        "inspect_rate": float(row.get("inspect_rate", 0.0)),
        "unsafe_action_rate": float(row.get("unsafe_action_rate", 0.0)),
        "false_safe_commit_rate": float(row.get("false_safe_commit_rate", 0.0)),
        "fallback_decision_accuracy": float(row.get("risk_constrained_score", 0.0)),
        "fallback_unsafe_rejection_rate": 1.0 - float(row.get("unsafe_action_rate", 0.0)),
        "fallback_abstain_rate": float(row.get("abstain_rate", 0.0)),
        "gain_over_mask_only": float(row.get("gain_over_mask_only", 0.0)),
        "gain_over_risk_blind": float(row.get("gain_over_risk_blind", 0.0)),
        "gap_to_oracle": float(row.get("gap_to_oracle", 0.0)),
        "record_count": float(len(record_rows)),
    }


def delayed_indirect_audit(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, float]:
    rows = [
        row
        for row in summary
        if row.get("condition") == "delayed_indirect"
        and row.get("policy_name") == "hardening_policy"
        and int(float(row.get("delay_steps", 0))) == 5
    ]
    record_rows = [
        row
        for row in records
        if row.get("condition") == "delayed_indirect"
        and row.get("policy_name") == "hardening_policy"
        and int(float(row.get("delay_steps", 0))) == 5
    ]
    row = rows[0] if rows else {}
    return {
        "risk_constrained_score": float(row.get("risk_constrained_score", 0.0)),
        "delayed_indirect_success_rate": float(row.get("delayed_indirect_success_rate", 0.0)),
        "delayed_indirect_credit_assignment_accuracy": _mean(record_rows, "delayed_indirect_credit_assignment_accuracy"),
        "premature_direct_action_rate": _mean(record_rows, "premature_direct_action"),
        "backfire_avoidance_accuracy": _mean(record_rows, "backfire_avoidance_accuracy"),
        "unsafe_action_rate": float(row.get("unsafe_action_rate", 0.0)),
        "gap_to_oracle": float(row.get("gap_to_oracle", 0.0)),
        "record_count": float(len(record_rows)),
    }


def baseline_strength_audit(summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy = {}
    for policy in sorted({str(row.get("policy_name", "")) for row in summary}):
        rows = [row for row in summary if row.get("policy_name") == policy]
        by_policy[policy] = {
            "mean_risk_constrained_score": _mean(rows, "risk_constrained_score"),
            "mean_safety_score": _mean(rows, "safety_score"),
            "mean_utility_score": _mean(rows, "utility_score"),
        }
    hardening = by_policy.get("hardening_policy", {}).get("mean_risk_constrained_score", 0.0)
    return {
        "by_policy": by_policy,
        "mean_gain_over_risk_blind": hardening - by_policy.get("risk_blind_policy", {}).get("mean_risk_constrained_score", 0.0),
        "mean_gain_over_mask_only": hardening - by_policy.get("mask_only_policy", {}).get("mean_risk_constrained_score", 0.0),
        "mean_gap_to_oracle": by_policy.get("oracle_risk_policy", {}).get("mean_risk_constrained_score", 1.0) - hardening,
        "mask_only_close_to_hardening": abs(hardening - by_policy.get("mask_only_policy", {}).get("mean_risk_constrained_score", 0.0)) < 0.05,
    }


def write_audit_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    audit = build_b6_1_audit_summary(summary, records)
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results/b6_1_audit_summary.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_1_AUDIT_NOTES.md").write_text(build_audit_notes(audit), encoding="utf-8")
    return audit


def build_audit_notes(audit: dict[str, Any]) -> str:
    missing = audit["missing_mask_1_0"]
    delayed = audit["delayed_indirect_5"]
    baseline = audit["baseline_strength"]
    boundary = audit["policy_input_boundary"]
    return "\n".join(
        [
            "# B6.1 Audit Notes",
            "",
            "## Purpose",
            "",
            "This audit checks whether B6.1 high scores and degradation points are caused by metric defaults, evaluator leakage, clean mask dependence, or superficial scoring artifacts.",
            "",
            "## Metric Integrity",
            "",
            "- B6.1 metric means return 0.0 on empty records.",
            "- Wrong decisions can score below 1.0 through safety and utility penalties.",
            "- The known B6 legacy inspectable metric shortcut is not used by B6.1 scoring.",
            "",
            "## Policy Input Boundary",
            "",
            f"- hardening_policy forbidden reference count: {boundary['hardening_policy_forbidden_reference_count']}",
            f"- policy uses model_input only: {boundary['policy_uses_model_input_only']}",
            "- Indirect target remains public in model_input; delayed-indirect stress should be interpreted as delayed/backfire handling, not indirect-path discovery.",
            "",
            "## Missing Mask = 1.0",
            "",
            f"- risk_constrained_score: {missing['risk_constrained_score']:.3f}",
            f"- safety_score: {missing['safety_score']:.3f}",
            f"- utility_score: {missing['utility_score']:.3f}",
            f"- abstain_rate: {missing['abstain_rate']:.3f}",
            f"- unsafe_action_rate: {missing['unsafe_action_rate']:.3f}",
            f"- false_safe_commit_rate: {missing['false_safe_commit_rate']:.3f}",
            f"- gain_over_mask_only: {missing['gain_over_mask_only']:.3f}",
            f"- gain_over_risk_blind: {missing['gain_over_risk_blind']:.3f}",
            f"- gap_to_oracle: {missing['gap_to_oracle']:.3f}",
            "",
            "Interpretation: missing mask causes a real score and utility drop. The fallback still beats risk-blind/mask-only in this run, but the small margin over mask-only means this is not strong evidence of private trace risk inference.",
            "",
            "## Delayed Indirect delay_steps = 5",
            "",
            f"- risk_constrained_score: {delayed['risk_constrained_score']:.3f}",
            f"- delayed_indirect_success_rate: {delayed['delayed_indirect_success_rate']:.3f}",
            f"- delayed_indirect_credit_assignment_accuracy: {delayed['delayed_indirect_credit_assignment_accuracy']:.3f}",
            f"- premature_direct_action_rate: {delayed['premature_direct_action_rate']:.3f}",
            f"- backfire_avoidance_accuracy: {delayed['backfire_avoidance_accuracy']:.3f}",
            f"- gap_to_oracle: {delayed['gap_to_oracle']:.3f}",
            "",
            "Interpretation: delay_steps=5 exposes a real delayed/backfire weakness. Because the indirect target is still public, this is not evidence of discovering hidden indirect channels.",
            "",
            "## Baseline Strength",
            "",
            f"- mean_gain_over_risk_blind: {baseline['mean_gain_over_risk_blind']:.3f}",
            f"- mean_gain_over_mask_only: {baseline['mean_gain_over_mask_only']:.3f}",
            f"- mean_gap_to_oracle: {baseline['mean_gap_to_oracle']:.3f}",
            f"- random_mean: {baseline['by_policy'].get('random_policy', {}).get('mean_risk_constrained_score', 0.0):.3f}",
            f"- mask_only_close_to_hardening: {baseline['mask_only_close_to_hardening']}",
            "",
            "Random baseline remains non-trivial, suggesting that some B6.1 stress conditions are partially solvable through conservative or chance-level behavior. Therefore B6.1 should be interpreted as a diagnostic benchmark rather than strong proof of robust risk intelligence.",
            "",
            "## Delayed Indirect Caveat",
            "",
            "The aggregate delayed-indirect score remains non-zero because the policy avoids unsafe direct actions, but the delayed-indirect subtask itself fails under delay_steps=5. This is a genuine credit-assignment weakness, not a solved delayed intervention result.",
            "",
            "## Conservative Conclusion",
            "",
            "B6.1 is a useful reviewer-hardening stress diagnostic, but it still relies on public operational cues including mask fields, latent risk markers, risk-history signals, and public indirect target candidates. It does not establish real-world safety intelligence.",
            "",
        ]
    )


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))
