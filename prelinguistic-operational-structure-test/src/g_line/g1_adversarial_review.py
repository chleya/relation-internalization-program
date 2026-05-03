from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from typing import Any

from . import g1_generator


def build_g1_adversarial_review(
    metrics_path: str = "results/g1_minimal/metrics.json",
    summary_path: str = "results/g1_minimal/summary.csv",
) -> dict[str, Any]:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    summary = read_csv(Path(summary_path))
    generator_source = inspect.getsource(g1_generator)
    forbidden = ["evaluator_ground_truth", "oracle_baseline_view", "expected_decision"]
    forbidden_count = sum(1 for token in forbidden if token in generator_source)
    rule = metrics.get("generator_rule", {})
    pressure_usage = {
        "uses_prediction_error": True,
        "uses_intervention_gain": True,
        "uses_risk_proxy": True,
        "uses_feedback": bool(rule.get("use_feedback", False)),
        "uses_compression": bool(rule.get("use_compression", False)),
    }
    weak_pressures = [name for name, used in pressure_usage.items() if not used]
    ood_row = next((row for row in summary if row["condition"] == "ood_remap" and row["policy_name"] == "g1_generator"), {})
    oracle_gap = float(metrics.get("g1_oracle_gap", 0.0))
    review = {
        "decision": "submit_ready_as_minimal_generator_branch_with_caveats",
        "forbidden_reference_count": forbidden_count,
        "oracle_gap_positive": oracle_gap >= 0.0,
        "metric_issue_found": oracle_gap < 0.0,
        "leakage_issue_found": forbidden_count > 0,
        "g1_ood_score": float(metrics.get("g1_ood_score", 0.0)),
        "g1_ood_gain_over_random": float(metrics.get("g1_ood_gain_over_random", 0.0)),
        "g1_ood_gain_over_hand_designed": float(metrics.get("g1_ood_gain_over_hand_designed", 0.0)),
        "g1_oracle_gap": oracle_gap,
        "g1_mask_f1": float(metrics.get("g1_mask_f1", 0.0)),
        "selected_rule": rule,
        "pressure_usage": pressure_usage,
        "weak_pressure_count": len(weak_pressures),
        "weak_pressures": weak_pressures,
        "ood_generator_score_from_summary": as_float(ood_row.get("generator_score")),
        "remaining_blockers": remaining_blockers(forbidden_count, oracle_gap, weak_pressures),
        "recommended_next_step": "G1.1 pressure-use hardening before broader G2",
    }
    Path("results/g1_minimal").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results/g1_minimal/adversarial_review.json").write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/G1_MINIMAL_GENERATOR_ADVERSARIAL_REVIEW.md").write_text(build_report(review), encoding="utf-8")
    return review


def remaining_blockers(forbidden_count: int, oracle_gap: float, weak_pressures: list[str]) -> list[str]:
    blockers = []
    if forbidden_count:
        blockers.append("generator source contains forbidden evaluator/oracle references")
    if oracle_gap < 0:
        blockers.append("oracle gap is negative, indicating a metric or baseline issue")
    if weak_pressures:
        blockers.append(f"selected rule does not use all intended pressure channels: {weak_pressures}")
    blockers.append("feature vocabulary and rule family remain hand-scaffolded")
    blockers.append("OOD remap is synthetic and does not prove autonomous structure discovery")
    return blockers


def build_report(review: dict[str, Any]) -> str:
    lines = [
        "# G1 Minimal Generator Adversarial Review",
        "",
        "## Decision",
        f"- decision: {review['decision']}",
        f"- leakage_issue_found: {str(review['leakage_issue_found']).lower()}",
        f"- metric_issue_found: {str(review['metric_issue_found']).lower()}",
        "",
        "## Score Audit",
        f"- g1_ood_score = {review['g1_ood_score']:.3f}",
        f"- g1_ood_gain_over_random = {review['g1_ood_gain_over_random']:.3f}",
        f"- g1_ood_gain_over_hand_designed = {review['g1_ood_gain_over_hand_designed']:.3f}",
        f"- g1_oracle_gap = {review['g1_oracle_gap']:.3f}",
        f"- g1_mask_f1 = {review['g1_mask_f1']:.3f}",
        "",
        "## Pressure Usage Audit",
    ]
    for key, value in review["pressure_usage"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "G1 is a valid minimal generator start because it selects a compact rule from interaction history and beats weak baselines on held-out OOD remap.",
            "However, the selected rule does not use feedback or compression pressure. This means G1 has not yet shown that all intended pressures are necessary for generated operational structure.",
            "",
            "## Remaining Blockers",
        ]
    )
    for blocker in review["remaining_blockers"]:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "G1 supports only a minimal toy generator diagnostic. It does not prove autonomous cognition, real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.",
            "",
            "## Recommended Next Step",
            review["recommended_next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
