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
            "# V5 Governance Shell Report",
            "",
            "## 1. Motivation",
            "",
            "V5 tests whether relation-chain review outputs can be logged, gated, replayed, and assigned to a responsibility chain.",
            "",
            "## 2. Gates",
            "",
            *[f"- `{key}` >= {value}" for key, value in gates.items()],
            "",
            "## 3. Results",
            "",
            markdown_table(rows),
            "",
            "## 4. Interpretation",
            "",
            "A governance shell passes only if it blocks autonomous approval, routes takeover correctly, records replayable logs, and preserves human responsibility.",
            "",
            "## 5. Claim Boundary",
            "",
            "Supported: toy governance-shell diagnostics for relation-chain review workflows.",
            "",
            "Unsupported: real governance, real engineering approval, legal responsibility automation, or deployment-ready workflow.",
            "",
        ]
    )


def build_self_audit(rows: list[dict[str, Any]]) -> str:
    passing = [row["shell"] for row in rows if float(row["gated_v5_score"]) > 0]
    return "\n".join(
        [
            "# V5 Self-Audit",
            "",
            "## What V5 Improves",
            "",
            "- Adds audit logs around review outputs.",
            "- Enforces no autonomous approval in toy cases.",
            "- Adds replay hashes for decision events.",
            "- Requires human responsibility traceability.",
            "",
            "## Passing Shells",
            "",
            ", ".join(passing) if passing else "None.",
            "",
            "## Remaining Weaknesses",
            "",
            "- This is a toy governance shell.",
            "- Logs are synthetic.",
            "- Replay hashes are not security guarantees.",
            "- No real legal, organizational, or engineering process is modeled.",
            "",
        ]
    )


def build_claims() -> str:
    return "\n".join(
        [
            "# V5 Claims",
            "",
            "## Supported",
            "",
            "- Relation-chain review outputs can be wrapped in a toy governance shell with logging, gates, replay, and responsibility traceability.",
            "- Auto-approval, missing logs, missing replay, and missing responsibility controls are rejected by gated metrics.",
            "",
            "## Not Supported",
            "",
            "- Real engineering governance.",
            "- Legal responsibility automation.",
            "- Deployment-ready approval workflow.",
            "- Real construction approval.",
            "",
        ]
    )


def build_limitations() -> str:
    return "\n".join(
        [
            "# V5 Limitations",
            "",
            "- V5 cases are synthetic.",
            "- Governance events are deterministic.",
            "- Replay hashes only check record consistency.",
            "- No authentication, access control, or legal process exists.",
            "- Passing V5 does not imply deployability.",
            "",
        ]
    )

