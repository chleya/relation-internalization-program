from __future__ import annotations

from typing import Any


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0].keys()) if rows else []
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# R1 Relation Internalization Agent Report",
            "",
            "## 1. Motivation",
            "",
            "R1 returns to the core architecture: a non-LLM agent learns usable relations through interaction.",
            "",
            "## 2. Results",
            "",
            markdown_table(rows),
            "",
            "## 3. Interpretation",
            "",
            "The relation agent must actively explore, recover internal links, act by relation simulation, answer counterfactuals, and respond to relation edits.",
            "",
            "## 4. Boundary",
            "",
            "R1 is a toy diagnostic. It is not a real engineering agent or general intelligence system.",
            "",
        ]
    )


def build_self_audit() -> str:
    return "\n".join(
        [
            "# R1 Self-Audit",
            "",
            "## What R1 Improves",
            "",
            "- Returns from review/governance shells to an agent body.",
            "- Uses active intervention rather than language generation.",
            "- Requires internal relation links, counterfactuals, edits, and action.",
            "",
            "## Remaining Weaknesses",
            "",
            "- World is small and deterministic.",
            "- Candidate relation links are predefined.",
            "- The relation learner is simple count-based induction.",
            "- This is not a real alternative to LLM-scale intelligence yet.",
            "",
        ]
    )


def build_claims() -> str:
    return "\n".join(
        [
            "# R1 Claims",
            "",
            "## Supported",
            "",
            "- A small non-LLM agent can learn usable relation links through interaction in this toy world.",
            "- The learned relation table can drive action, counterfactuals, and internal edits.",
            "",
            "## Not Supported",
            "",
            "- General intelligence.",
            "- Real engineering competence.",
            "- Unrestricted relation discovery.",
            "- Replacement for LLMs.",
            "",
        ]
    )


def build_limitations() -> str:
    return "\n".join(
        [
            "# R1 Limitations",
            "",
            "- Toy process world only.",
            "- Predefined candidate links.",
            "- No neural perception.",
            "- No language understanding.",
            "- No open-ended environment.",
            "",
        ]
    )

