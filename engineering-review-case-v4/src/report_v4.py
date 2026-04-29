from __future__ import annotations

from typing import Any


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "agent",
        "relation_chain_specificity",
        "action_point_mapping",
        "uncertainty_takeover_quality",
        "verification_indicator_quality",
        "responsibility_boundary_quality",
        "unsafe_review_rejection",
        "gated_v4_score",
    ]
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
            "# V4 Engineering Review Report",
            "",
            "## 1. Motivation",
            "",
            "V4 tests whether relation-chain diagnostics can be expressed as a bounded engineering-review workflow.",
            "",
            "## 2. Boundary",
            "",
            "This is a toy diagnostic. It is not a real geotechnical safety model and does not approve real work.",
            "",
            "## 3. Metrics And Gates",
            "",
            *[f"- `{key}` >= {value}" for key, value in gates.items()],
            "",
            "## 4. Results",
            "",
            markdown_table(rows),
            "",
            "## 5. Interpretation",
            "",
            "A reviewer passes only when it exposes relation chains, action points, uncertain links, verification indicators, takeover conditions, and responsibility boundaries.",
            "",
            "## 6. Failure Cases",
            "",
            "Negative controls should have `gated_v4_score = 0.0`. If they pass, the V4 gates are too weak.",
            "",
            "## 7. Claim Boundary",
            "",
            "Supported: bounded toy review-case diagnostics for relation-chain audit.",
            "",
            "Unsupported: real geotechnical modeling, real safety prediction, or deployment-ready engineering review.",
            "",
        ]
    )


def build_self_audit(rows: list[dict[str, Any]]) -> str:
    passing = [row["agent"] for row in rows if float(row["gated_v4_score"]) > 0]
    return "\n".join(
        [
            "# V4 Self-Audit",
            "",
            "## What V4 Improves",
            "",
            "- Moves the relation-chain diagnostic into a bounded review workflow.",
            "- Requires uncertainty and takeover audit, not just engineering-sounding text.",
            "- Requires a responsibility boundary and explicit non-deployment claim.",
            "- Keeps generic and surface-warning reviewers as negative controls.",
            "",
            "## Passing Reviewers",
            "",
            ", ".join(passing) if passing else "None.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Cases are curated toy cases.",
            "- Relation links remain predefined.",
            "- Scoring is rule-based and can be gamed by a formatter.",
            "- No real monitoring uncertainty model is used.",
            "- No engineering code, design standard, or expert validation is included.",
            "",
            "## False Positive Risks",
            "",
            "- A reviewer may learn the fixed schema rather than a general review skill.",
            "- Audit strings may be correct by construction.",
            "- Negative controls may be too weak if future cases become richer.",
            "",
            "## Boundary Statement",
            "",
            "This remains a toy diagnostic, not a real geotechnical time-series model, not unrestricted temporal relation discovery, and not deployment-ready engineering AI.",
            "",
        ]
    )


def build_claims() -> str:
    return "\n".join(
        [
            "# V4 Claims",
            "",
            "## Supported",
            "",
            "- The project can express relation-chain diagnostics as bounded toy engineering review cases.",
            "- Generic review text is rejected when it lacks relation-chain, uncertainty, takeover, and responsibility-boundary content.",
            "- The review workflow makes the human takeover boundary explicit.",
            "",
            "## Not Supported",
            "",
            "- Real geotechnical correctness.",
            "- Real construction-plan approval.",
            "- Safety-calibrated slope monitoring.",
            "- Deployment-ready engineering review AI.",
            "- Unrestricted relation discovery.",
            "",
        ]
    )


def build_limitations() -> str:
    return "\n".join(
        [
            "# V4 Limitations",
            "",
            "- V4 uses curated toy cases, not real project records.",
            "- The relation chain is predefined.",
            "- The reviewers are deterministic baselines, not deployed AI systems.",
            "- The scoring functions are diagnostic checks, not engineering standards.",
            "- Passing V4 does not imply safety, correctness, or expert approval.",
            "",
            "Best interpretation:",
            "",
            "```text",
            "V4 tests whether an engineering-style review exposes usable relation chains and takeover boundaries in a toy setting.",
            "```",
            "",
        ]
    )

