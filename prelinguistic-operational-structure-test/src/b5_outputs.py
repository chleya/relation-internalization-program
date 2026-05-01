from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .b5_closed_loop_metrics import B5_SUMMARY_KEYS


B5_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "episode_type",
    "needs_inspection",
    "predicted_inspect_region",
    "oracle_inspect_region",
    "inspect_skipped",
    "epistemic_gain",
    "inspection_cost",
    "trace_before_region",
    "trace_after_inspection_region",
    "trace_update_correct",
    "predicted_intervention_action_type",
    "predicted_intervention_region",
    "oracle_intervention_action_type",
    "oracle_intervention_region",
    "pragmatic_gain",
    "intervention_cost",
    "consequence_value",
    "trace_after_feedback_region",
    "feedback_revision_correct",
    "closed_loop_value",
    "baseline_name",
    "baseline_value",
    "planning_budget_used",
    "gate_pass",
    "note",
]


def write_b5_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b5_closed_loop_summary.csv"), summary, B5_SUMMARY_KEYS)
    write_csv(Path("results/b5_closed_loop_records.csv"), records, B5_RECORD_KEYS)
    write_csv(Path("results/b5_epistemic_pragmatic_values.csv"), [row for row in records if row.get("record_kind") == "closed_loop_policy"])
    write_csv(Path("results/b5_trace_update_results.csv"), [row for row in records if row.get("record_kind") == "closed_loop_policy"])
    write_csv(Path("results/b5_feedback_revision_results.csv"), [row for row in records if row.get("record_kind") == "closed_loop_policy"])
    write_csv(Path("results/b5_baseline_comparison.csv"), [row for row in records if row.get("record_kind") == "baseline"])
    write_csv(Path("results/b5_planning_budget_results.csv"), [row for row in records if row.get("record_kind") == "closed_loop_policy"])
    Path("reports/B5_EPISTEMIC_PRAGMATIC_CLOSED_LOOP.md").write_text(build_b5_report(summary), encoding="utf-8")
    Path("reports/B5_CLOSED_LOOP_SELF_AUDIT.md").write_text(build_b5_self_audit(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_b5_report(summary: list[dict[str, Any]]) -> str:
    passed = [row["model"] for row in summary if float(row.get("b5_closed_loop_score", 0.0)) > 0.0]
    interpretation = (
        "B5 supports that, in the toy PLOS environment, private delayed operational trace can support a minimal epistemic-pragmatic closed loop: the system can decide when to inspect, update trace from inspection, intervene based on updated trace, observe consequence, and revise trace under budget."
        if passed
        else "B5 shows that current private delayed traces support one-shot inspection and intervention, but not yet closed-loop operational update."
    )
    return "\n".join(
        [
            "# B5 Epistemic-Pragmatic Closed-Loop Operation",
            "",
            "## 1. Purpose",
            "",
            "B5 tests whether B4.2 private delayed traces can support a two-step closed loop: observe -> inspect -> update trace -> intervene -> observe consequence -> revise trace.",
            "",
            "## 2. Background",
            "",
            "PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: inspection degeneracy audit. B3.2: mechanism-disambiguating active inspection. B4: trace-guided intervention. B4.1: fixed action-type shortcut discovered. B4.2: action-type disambiguation. B5: epistemic-pragmatic closed-loop operation.",
            "",
            "## 3. Closed-loop Task",
            "",
            "- observe",
            "- inspect or skip",
            "- update trace",
            "- intervene or skip",
            "- observe consequence",
            "- revise trace",
            "",
            "## 4. Epistemic vs Pragmatic Value",
            "",
            "- epistemic value = information gain from inspection",
            "- pragmatic value = outcome improvement from intervention",
            "",
            "## 5. Baselines",
            "",
            "- random",
            "- saliency",
            "- short-horizon",
            "- inspect-always",
            "- intervene-immediately",
            "- oracle",
            "",
            "## 6. Results",
            "",
            markdown_table(summary),
            "",
            "## 7. Interpretation",
            "",
            interpretation,
            "",
            "## 8. Claim Boundary",
            "",
            "Do not claim real control.",
            "Do not claim robotics capability.",
            "Do not claim engineering deployment.",
            "Do not claim human-like active inference.",
            "Do not claim language-free cognition solved.",
            "",
        ]
    )


def build_b5_self_audit() -> str:
    return "\n".join(
        [
            "# B5 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Moves from one-shot action to closed-loop operation.",
            "- Separates epistemic and pragmatic value.",
            "- Tests trace update after inspection.",
            "- Tests intervention after updated trace.",
            "- Tests feedback revision after consequence.",
            "- Adds inspect-vs-intervene timing.",
            "- Adds planning budget.",
            "- Adds inspect-always / intervene-immediately baselines.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Only two-step closed loop.",
            "- Inspection and intervention values are simulator-defined.",
            "- Trace update remains evaluator-designed.",
            "- No real robot control.",
            "- No real engineering environment.",
            "- Passing does not prove general active intelligence.",
            "",
            "## False Positive Risks",
            "",
            "- Model may always inspect first.",
            "- Model may always intervene immediately.",
            "- Trace update may be a direct oracle-like patch.",
            "- Feedback revision may be scripted.",
            "- Epistemic/pragmatic values may leak evaluator assumptions.",
            "- Planning budget may be too loose.",
            "- Baselines may be too weak.",
            "",
            "## Required Failure Checks",
            "",
            "1. inspect-always baseline matches model",
            "2. intervene-immediately baseline matches model",
            "3. trace update does not improve intervention",
            "4. inspection does not reduce uncertainty",
            "5. feedback revision does not change trace",
            "6. planning budget is exceeded",
            "7. oracle closed-loop score is low",
            "8. random baseline passes",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(B5_SUMMARY_KEYS) + " |", "| " + " | ".join("---" for _ in B5_SUMMARY_KEYS) + " |"]
    for row in rows:
        values = []
        for key in B5_SUMMARY_KEYS:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
