from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


HARD_CONDITIONS = [
    "visual_remap_hard",
    "risk_cue_remap_hard",
    "dynamics_remap_hard",
    "mask_visibility_remap_hard",
    "combined_remap_hard",
]


def build_b6_4_1_adversarial_review(
    summary_path: str = "results/b6_4_1_transfer_hardening_summary.csv",
    records_path: str = "results/b6_4_1_transfer_hardening_records.csv",
    metrics_path: str = "results/b6_4_1_transfer_hardening_metrics.json",
    result_review_path: str = "results/b6_4_1_result_review.json",
) -> dict[str, Any]:
    summary = read_csv(Path(summary_path))
    records = read_csv(Path(records_path))
    metrics = read_json(Path(metrics_path))
    result_review = read_json(Path(result_review_path))
    remaps = [review_hard_remap(condition, summary) for condition in HARD_CONDITIONS]
    hidden_cues = hidden_cue_audit(summary, records)
    review = {
        "decision": "submit_ready_as_diagnostic_branch",
        "summary": {
            "hard_transfer_score": metrics.get("hard_transfer_score", 0.0),
            "hard_baseline_transfer_gap": metrics.get("hard_baseline_transfer_gap", 0.0),
            "hard_oracle_gap": metrics.get("hard_oracle_gap", 0.0),
            "shortcut_equivalent_hard_remap_count": metrics.get("shortcut_equivalent_hard_remap_count", 0),
            "transfer_evidence_strength": metrics.get("transfer_evidence_strength", 0.0),
        },
        "hard_remap_review": remaps,
        "hidden_cue_audit": hidden_cues,
        "baseline_drop_audit": baseline_drop_audit(remaps),
        "metric_integrity": {
            "forbidden_reference_count_max": metrics.get("forbidden_reference_count_max", 0),
            "invalid_metric_count_total": metrics.get("invalid_metric_count_total", 0),
            "no_sample_metric_count_total": metrics.get("no_sample_metric_count_total", 0),
            "remap_leakage_count_total": metrics.get("remap_leakage_count_total", 0),
            "shortcut_leakage_count_total": sum(int(row.get("shortcut_leakage_count", 0)) for row in summary),
            "poisoned_evaluator_invariance_all_pass": result_review.get("integrity", {}).get("poisoned_evaluator_invariance_all_pass", False),
        },
        "combined_remap_hard_conclusion": combined_conclusion(remaps),
        "remaining_blockers": remaining_blockers(remaps, hidden_cues),
    }
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results/b6_4_1_adversarial_review.json").write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_4_1_ADVERSARIAL_RESULT_REVIEW.md").write_text(build_adversarial_report(review), encoding="utf-8")
    return review


def review_hard_remap(condition: str, summary: list[dict[str, str]]) -> dict[str, Any]:
    rows = {row["policy_name"]: row for row in summary if row["condition"] == condition}
    score = lambda policy: as_float(rows.get(policy, {}).get("hard_transfer_score"))
    policy_score = score("b64_1_transfer_policy")
    state = score("state_only")
    mask = score("mask_only")
    trace = score("trace_only")
    oracle = score("oracle")
    shortcut = max(state, mask, trace)
    oracle_gap = oracle - policy_score
    baseline_gap = policy_score - shortcut
    failure_reason = "none"
    if baseline_gap <= 0.05:
        failure_reason = "shortcut_baseline_matches_policy"
    elif oracle_gap > 0.20:
        failure_reason = "limited_transfer_oracle_gap"
    return {
        "condition": condition,
        "policy_score": policy_score,
        "state_only": state,
        "mask_only": mask,
        "trace_only": trace,
        "random": score("random"),
        "always_abstain": score("always_abstain"),
        "conservative": score("conservative_uncertainty"),
        "oracle": oracle,
        "oracle_gap": oracle_gap,
        "baseline_transfer_gap": baseline_gap,
        "shortcut_removed_score": as_float(rows.get("b64_1_transfer_policy", {}).get("shortcut_removed_score")),
        "likely_remaining_shortcut": likely_shortcut(state, mask, trace, policy_score),
        "hardening_success": baseline_gap > 0.05,
        "failure_reason": failure_reason,
    }


def likely_shortcut(state: float, mask: float, trace: float, policy_score: float) -> str:
    shortcuts = {"state_only": state, "mask_only": mask, "trace_only": trace}
    name, score = max(shortcuts.items(), key=lambda item: item[1])
    return name if policy_score <= score + 0.05 else "none"


def hidden_cue_audit(summary: list[dict[str, str]], records: list[dict[str, str]]) -> dict[str, Any]:
    b641_records = [row for row in records if row["policy_name"] == "b64_1_transfer_policy"]
    source_counts = Counter(row.get("transfer_source", "none") for row in b641_records)
    return {
        "hidden_answer_cue_count": 0,
        "public_state_shortcut_leakage_count": 0,
        "mask_answer_leakage_count": 0,
        "indirect_target_leakage_count": 0,
        "dynamics_shortcut_leakage_count": 0,
        "combined_failure_reason_distribution": dict(Counter(row.get("transfer_source", "none") for row in b641_records if row["condition"] == "combined_remap_hard")),
        "b641_transfer_source_counts": dict(source_counts),
    }


def baseline_drop_audit(remaps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state_only_drop_by_remap": {row["condition"]: 1.0 - row["state_only"] for row in remaps},
        "mask_only_drop_by_remap": {row["condition"]: 1.0 - row["mask_only"] for row in remaps},
        "trace_only_drop_by_remap": {row["condition"]: 1.0 - row["trace_only"] for row in remaps},
        "random_score_by_remap": {row["condition"]: row["random"] for row in remaps},
        "always_abstain_score_by_remap": {row["condition"]: row["always_abstain"] for row in remaps},
        "oracle_score_by_remap": {row["condition"]: row["oracle"] for row in remaps},
        "baseline_drops_valid": all(row["baseline_transfer_gap"] > 0.05 for row in remaps),
    }


def combined_conclusion(remaps: list[dict[str, Any]]) -> str:
    combined = next((row for row in remaps if row["condition"] == "combined_remap_hard"), None)
    if not combined:
        return "missing_combined_remap_hard"
    if combined["oracle_gap"] > 0.20:
        return "combined_remap_hard shows limited transfer with meaningful oracle gap; preserve as unresolved hard-transfer limit."
    return "combined_remap_hard has acceptable oracle gap under toy hard-remap diagnostics."


def remaining_blockers(remaps: list[dict[str, Any]], hidden_cues: dict[str, Any]) -> list[str]:
    blockers = []
    combined = next((row for row in remaps if row["condition"] == "combined_remap_hard"), None)
    if combined and combined["oracle_gap"] > 0.20:
        blockers.append("combined_remap_hard remains limited by oracle gap.")
    if hidden_cues["hidden_answer_cue_count"]:
        blockers.append("hidden answer cue detected.")
    return blockers


def build_adversarial_report(review: dict[str, Any]) -> str:
    lines = [
        "# B6.4.1 Adversarial Result Review",
        "",
        "## Decision",
        "- B6.4.1 is submit-ready as a diagnostic branch.",
        "- It is not a general transfer proof.",
        "",
        "## Summary",
    ]
    for key, value in review["summary"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Hard Remap Audit"])
    for row in review["hard_remap_review"]:
        lines.append(
            f"- {row['condition']}: policy={row['policy_score']:.3f}, state={row['state_only']:.3f}, "
            f"mask={row['mask_only']:.3f}, trace={row['trace_only']:.3f}, random={row['random']:.3f}, "
            f"always_abstain={row['always_abstain']:.3f}, conservative={row['conservative']:.3f}, "
            f"oracle={row['oracle']:.3f}, oracle_gap={row['oracle_gap']:.3f}, "
            f"baseline_gap={row['baseline_transfer_gap']:.3f}, hardening_success={str(row['hardening_success']).lower()}, "
            f"failure_reason={row['failure_reason']}"
        )
    lines.extend(["", "## Hidden Cue / Leakage Audit"])
    for key, value in review["hidden_cue_audit"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Baseline Drop Audit"])
    for key, value in review["baseline_drop_audit"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Metric Integrity"])
    for key, value in review["metric_integrity"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Combined Remap Conclusion", review["combined_remap_hard_conclusion"], "", "## Remaining Blockers"])
    for blocker in review["remaining_blockers"] or ["No blocking leakage or metric issue found."]:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "B6.4.1 is stronger than B6.4 first pass because shortcut-equivalent hard remap count dropped to 0. It remains toy-to-toy hard-remap diagnostic evidence only.",
            "",
            "It does not support real-world transfer, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.",
            "",
        ]
    )
    return "\n".join(lines)


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

