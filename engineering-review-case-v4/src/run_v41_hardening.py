from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .metrics_v41_hardening import evaluate_hardening


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
            "# V4.1 Review Hardening Report",
            "",
            "## 1. Motivation",
            "",
            "V4.1 attacks false positives in the V4 review-case diagnostic.",
            "",
            "## 2. Attacks",
            "",
            "- schema-template shortcut",
            "- fluent but non-specific review text",
            "- case-order memorization",
            "- responsibility-boundary boilerplate",
            "- unsafe approval phrasing",
            "",
            "## 3. Hardening Gates",
            "",
            *[f"- `{key}` >= {value}" for key, value in gates.items()],
            "",
            "## 4. Results",
            "",
            markdown_table(rows),
            "",
            "## 5. Interpretation",
            "",
            "`hardening_v41_gated_score` is zero unless the reviewer passes the original V4 gates and all V4.1 hardening gates.",
            "",
            "## 6. Claim Boundary",
            "",
            "Supported: bounded toy review diagnostics can be hardened against tested schema and prose shortcuts.",
            "",
            "Unsupported: real geotechnical safety review, real approval, or deployment-ready engineering AI.",
            "",
        ]
    )


def build_self_audit() -> str:
    return "\n".join(
        [
            "# V4.1 Self-Audit",
            "",
            "## What V4.1 Improves",
            "",
            "- Rejects schema-only review outputs.",
            "- Rejects fluent but non-specific engineering prose.",
            "- Tests case-order memorization by shuffled case order.",
            "- Rejects responsibility-boundary boilerplate without relation content.",
            "- Rejects unsafe plain approval phrasing.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Cases are still curated toy cases.",
            "- Relation links are still predefined.",
            "- A reviewer can still overfit the finite schema and finite case style.",
            "- The hardening metrics are rule-based diagnostics, not engineering standards.",
            "",
            "## Boundary Statement",
            "",
            "This remains a toy diagnostic, not real geotechnical review and not deployment-ready engineering AI.",
            "",
        ]
    )


def build_claims() -> str:
    return "\n".join(
        [
            "# V4.1 Claims",
            "",
            "## Supported",
            "",
            "- V4.1 rejects tested schema-template, non-specific prose, case-order, boilerplate, and unsafe-approval false positives.",
            "- The uncertainty-aware review remains the only passing reviewer in the current toy diagnostic.",
            "",
            "## Not Supported",
            "",
            "- Real engineering plan approval.",
            "- Safety-calibrated geotechnical review.",
            "- Unrestricted relation discovery.",
            "- Deployment-ready engineering AI.",
            "",
        ]
    )


def build_limitations() -> str:
    return "\n".join(
        [
            "# V4.1 Limitations",
            "",
            "- V4.1 hardening attacks are finite and hand-designed.",
            "- Passing V4.1 may still reflect adaptation to the toy schema.",
            "- It does not use real engineering documents or expert labels.",
            "- It does not validate numerical slope mechanics.",
            "",
        ]
    )


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rows, records = evaluate_hardening(config)
    outputs = config["outputs"]
    write_csv(outputs["summary"], rows)
    write_csv(outputs["records"], records)
    Path(outputs["report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["report"]).write_text(build_report(rows, config["hardening_gates"]), encoding="utf-8")
    Path(outputs["self_audit"]).write_text(build_self_audit(), encoding="utf-8")
    Path(outputs["claims"]).write_text(build_claims(), encoding="utf-8")
    Path(outputs["limitations"]).write_text(build_limitations(), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v41_hardening.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['agent']}: hardening_v41_gated_score={row['hardening_v41_gated_score']:.3f}")


if __name__ == "__main__":
    main()

