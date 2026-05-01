from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


B5_CLEAN_SUMMARY_KEYS = [
    "model",
    "seed",
    "clean_b5_closed_loop_score",
    "original_b5_closed_loop_score",
    "score_drop_from_original",
    "inspect_timing_accuracy",
    "epistemic_value_alignment",
    "trace_update_accuracy",
    "post_inspection_intervention_accuracy",
    "pragmatic_value_alignment",
    "feedback_revision_accuracy",
    "planning_budget_compliance",
    "clean_random_closed_loop_score",
    "clean_oracle_closed_loop_score",
    "model_input_leakage_count",
    "policy_output_oracle_usage_rate",
    "evaluator_ground_truth_policy_access_count",
    "oracle_baseline_access_violation_count",
    "clean_b5_pass",
]

B5_CLEAN_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "episode_type",
    "policy_input_sanitized",
    "predicted_inspect_region",
    "predicted_intervention_action_type",
    "predicted_intervention_region",
    "trace_update_region",
    "feedback_revision_region",
    "closed_loop_value",
    "gate_pass",
    "note",
]

B5_CLEAN_LEAKAGE_KEYS = [
    "episode_id",
    "model",
    "stage",
    "source",
    "leakage_count",
    "forbidden_key_paths",
    "fail_fast_triggered",
]


def write_b5_clean_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], leakage_records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b5_clean_closed_loop_summary.csv"), summary, B5_CLEAN_SUMMARY_KEYS)
    write_csv(Path("results/b5_clean_closed_loop_records.csv"), records, B5_CLEAN_RECORD_KEYS)
    write_csv(Path("results/b5_clean_leakage_audit.csv"), leakage_records, B5_CLEAN_LEAKAGE_KEYS)
    Path("reports/B5_CLEAN_ORACLE_FREE_RERUN.md").write_text(build_clean_report(summary), encoding="utf-8")
    Path("reports/B5_CLEAN_SELF_AUDIT.md").write_text(build_clean_self_audit(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_clean_report(summary: list[dict[str, Any]]) -> str:
    passed = any(float(row.get("clean_b5_pass", 0.0)) > 0.0 for row in summary)
    interpretation = (
        "B5-Clean fixes the oracle/value leakage found by B5.1. Under sanitized model inputs and evaluator-only ground truth, B5 remains a valid oracle-free closed-loop path diagnostic in the toy PLOS environment."
        if passed
        else "B5-Clean shows that the original B5 result is not trustworthy as closed-loop evidence because oracle/value leakage remains or clean rerun collapses."
    )
    return "\n".join(
        [
            "# B5-Clean Oracle-Free Closed-Loop Rerun",
            "",
            "## 1. Purpose",
            "",
            "B5.1 found value/oracle leakage in B5 policy input. B5-Clean fixes the experimental hygiene by separating model_input, evaluator_ground_truth, and oracle_baseline_view.",
            "",
            "## 2. Problem Found by B5.1",
            "",
            "- value_leakage_count = 800.000",
            "- scripted_update_score = 1.000",
            "- model_gain_over_scripted_update = 0.000",
            "- feedback_revision_over_scripted_ratio = 1.000",
            "- cross_model_exact_plan_match_rate = 1.000",
            "",
            "## 3. Clean Separation",
            "",
            "- model_input: policy-visible only",
            "- evaluator_ground_truth: metrics-only",
            "- oracle_baseline_view: oracle baseline only",
            "",
            "## 4. Clean B5 Results",
            "",
            markdown_table(summary, B5_CLEAN_SUMMARY_KEYS),
            "",
            "## 5. Interpretation",
            "",
            interpretation,
            "",
            "## 6. Claim Boundary",
            "",
            "Do not claim real control.",
            "Do not claim robotics ability.",
            "Do not claim engineering deployment.",
            "Do not claim human-like active inference.",
            "Do not claim language-free cognition solved.",
            "",
        ]
    )


def build_clean_self_audit() -> str:
    return "\n".join(
        [
            "# B5-Clean Self-Audit",
            "",
            "## What This Fixes",
            "",
            "- Separates model_input from evaluator_ground_truth.",
            "- Restricts oracle fields to oracle baseline only.",
            "- Adds recursive forbidden-key leakage checks.",
            "- Adds fail-fast leakage guards.",
            "- Adds clean B5 rerun.",
            "- Adds clean B5.1 rerun.",
            "",
            "## What This Does Not Fix",
            "",
            "- It does not prove adaptive closed-loop structure.",
            "- It does not solve scripted update degeneracy by itself.",
            "- It does not solve cross-model plan overlap by itself.",
            "- It does not add new model capability.",
            "- It does not prove natural emergence.",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
