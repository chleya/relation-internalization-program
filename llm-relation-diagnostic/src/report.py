from __future__ import annotations

from typing import Any


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(summary: list[dict[str, Any]]) -> str:
    result_keys = set(summary[0].keys()) if summary else set()
    if "budgeted_inspect" in result_keys and "random_symbol_transfer" not in result_keys:
        return "\n".join(
            [
                "# Budgeted Inspect Stress Report",
                "",
                "## 1. Purpose",
                "",
                "This stage isolates the action-guiding `budgeted_inspect` gate. It tests whether a solver can choose the variable whose inspection makes a relation-chain outcome verifiable under a one-inspection budget.",
                "",
                "The stage is a false-positive diagnostic. Passing it would not prove general relation understanding.",
                "",
                "## 2. Results",
                "",
                markdown_table(summary),
                "",
                "## 3. Interpretation",
                "",
                "- `relation_oracle` is a sanity-check upper bound.",
                "- `missingness_template` should fail when the first missing variable is irrelevant or when the chain is already verifiable.",
                "- `surface_audit` should fail when exact inspect selection and exact audit-link selection are required.",
                "",
                "## 4. Boundary",
                "",
                "This is a toy black-box diagnostic. It is not evidence of deployment-ready engineering judgment, unrestricted causal discovery, or LLM relation understanding.",
                "",
            ]
        )
    if "local_edit_locality" in result_keys and "random_symbol_transfer" not in result_keys:
        return "\n".join(
            [
                "# Local Edit Behavior Stress Report",
                "",
                "## 1. Purpose",
                "",
                "This stage isolates the behavior-level `local_edit_locality` gate. It tests whether a solver can apply one support-specific relation edit while preserving other support and unrelated-chain behavior.",
                "",
                "The stage is a false-positive diagnostic. Passing it would not prove general relation understanding.",
                "",
                "## 2. Results",
                "",
                markdown_table(summary),
                "",
                "## 3. Interpretation",
                "",
                "- `relation_oracle` is a sanity-check upper bound.",
                "- `edit_compliance` should fail if it treats edit acknowledgement as global behavior change.",
                "- `edit_no_behavior` should fail if it identifies the edited query but leaves target behavior unchanged.",
                "- A passing solver must return correct post-edit answers for target, other-support, and unrelated queries.",
                "",
                "## 4. Boundary",
                "",
                "This is a toy black-box diagnostic. It is not evidence of deployment-ready engineering judgment, unrestricted causal discovery, or LLM relation understanding.",
                "",
            ]
        )
    if "audit_correctness" in result_keys and "random_symbol_transfer" not in result_keys:
        return "\n".join(
            [
                "# Audit Correctness Stress Report",
                "",
                "## 1. Purpose",
                "",
                "This stage isolates the `audit_correctness` gate. It tests whether a solver can identify the exact unverifiable relation link, the right inspection variable, and the absence of an audit link when the missing variable is irrelevant.",
                "",
                "The stage is a false-positive diagnostic. Passing it would not prove general relation understanding.",
                "",
                "## 2. Results",
                "",
                markdown_table(summary),
                "",
                "## 3. Interpretation",
                "",
                "- `relation_oracle` is a sanity-check upper bound.",
                "- `surface_audit` should fail if it emits generic uncertainty text.",
                "- `missingness_template` should fail if it audits the first missing variable without checking relation relevance.",
                "- `reverse_audit` should fail if it names a related link in the wrong direction.",
                "- `outcome_audit` should fail if it defaults to inspecting the outcome instead of the blocking chain variable.",
                "",
                "## 4. Boundary",
                "",
                "This is a toy black-box diagnostic. It is not evidence of deployment-ready engineering judgment, unrestricted causal discovery, or LLM relation understanding.",
                "",
            ]
        )
    return "\n".join(
        [
            "# LLM Relation Diagnostic Report",
            "",
            "## 1. Purpose",
            "",
            "This stage evaluates black-box LLM outputs against relation-internalization gates using random symbols, support-conditioned rules, edits, audits, and inspection constraints.",
            "",
            "The stage is a false-positive diagnostic. Passing it would not prove general relation understanding.",
            "",
            "## 2. Results",
            "",
            markdown_table(summary),
            "",
            "## 3. Interpretation",
            "",
            "- `relation_oracle` is a sanity-check upper bound.",
            "- `global_mapping` should fail support-conditioned binding if it ignores support context.",
            "- `edit_compliance` should fail local edit locality if it treats edit acknowledgement as global behavior change.",
            "- `missingness_template` should fail noncritical missingness or budgeted inspect selection if it inspects any missing variable.",
            "- `surface_audit` should fail exact audit-link scoring if it emits generic uncertainty text.",
            "",
            "## 4. Boundary",
            "",
            "This is a toy black-box diagnostic. It is not evidence of deployment-ready engineering judgment, unrestricted causal discovery, or LLM relation understanding.",
            "",
        ]
    )


def build_self_audit(summary: list[dict[str, Any]]) -> str:
    passing = [row["solver"] for row in summary if row["llm_relation_gated_score"] > 0.0]
    return "\n".join(
        [
            "# LLM Relation Diagnostic Self-Audit",
            "",
            "## What This Adds",
            "",
            "- Defines a black-box JSON contract for LLM relation answers.",
            "- Separates support-conditioned relation binding from global symbol mapping.",
            "- Separates local edit behavior from edit acknowledgement.",
            "- Separates relation-specific uncertainty from generic missingness templates.",
            "- Keeps live LLM execution optional so the scoring layer can be tested without model noise.",
            "",
            "## Current Passing Solvers",
            "",
            ", ".join(passing) if passing else "None.",
            "",
            "## Remaining Weaknesses",
            "",
            "- The task family is small and synthetic.",
            "- Live model prompts may need hardening against invalid JSON.",
            "- The scoring layer checks black-box behavior only; it does not inspect representations.",
            "- A real LLM run should include multiple seeds and multiple local models before any claim is made.",
            "",
        ]
    )
