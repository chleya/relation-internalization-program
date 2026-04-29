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


def build_report(rows: list[dict[str, Any]], gates: dict[str, float]) -> str:
    return "\n".join(
        [
            "# V6 Multi-Party Audit Resolution Report",
            "",
            "## 1. Motivation",
            "",
            "V6 tests whether conflicting review outputs are preserved, compared, and routed to human resolution.",
            "",
            "## 2. Gates",
            "",
            *[f"- `{key}` >= {value}" for key, value in gates.items()],
            "",
            "## 3. Results",
            "",
            markdown_table(rows),
            "",
            "## 4. Boundary",
            "",
            "V6 is a toy audit-resolution diagnostic, not real dispute resolution or engineering approval.",
            "",
        ]
    )


def build_self_audit(rows: list[dict[str, Any]]) -> str:
    passing = [row["resolver"] for row in rows if float(row["gated_v6_score"]) > 0]
    return "\n".join(
        [
            "# V6 Self-Audit",
            "",
            "## Passing Resolvers",
            "",
            ", ".join(passing) if passing else "None.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Cases are synthetic.",
            "- Disagreement types are hand-designed.",
            "- Human resolution is represented as a route label only.",
            "- No real expert arbitration or legal process is modeled.",
            "",
        ]
    )


def build_claims() -> str:
    return "\n".join(
        [
            "# V6 Claims",
            "",
            "## Supported",
            "",
            "- Multi-review disagreement can be tested in toy audit-resolution cases.",
            "- Majority vote, confidence-only selection, auto compromise, minority-risk erasure, and missing audit trails are rejected.",
            "",
            "## Not Supported",
            "",
            "- Real engineering arbitration.",
            "- Legal dispute resolution.",
            "- Deployment-ready governance.",
            "",
        ]
    )


def build_limitations() -> str:
    return "\n".join(
        [
            "# V6 Limitations",
            "",
            "- V6 cases are synthetic.",
            "- Review disagreements are manually constructed.",
            "- The human resolution route is not a real process.",
            "- Passing V6 does not imply real-world dispute resolution ability.",
            "",
        ]
    )

