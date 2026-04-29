from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .metrics_v61_hardening import evaluate_hardening


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
            "# V6.1 Audit Resolution Hardening Report",
            "",
            "## 1. Motivation",
            "",
            "V6.1 attacks false positives in multi-party audit resolution.",
            "",
            "## 2. Attacks",
            "",
            "- fake evidence comparison",
            "- human-route label without responsibility boundary",
            "- hidden auto resolution",
            "- disagreement logged but minority risk omitted",
            "- tampered resolution hash",
            "",
            "## 3. Gates",
            "",
            *[f"- `{key}` >= {value}" for key, value in gates.items()],
            "",
            "## 4. Results",
            "",
            markdown_table(rows),
            "",
            "## 5. Boundary",
            "",
            "V6.1 remains a toy audit-resolution diagnostic, not real arbitration.",
            "",
        ]
    )


def build_self_audit() -> str:
    return "\n".join(
        [
            "# V6.1 Self-Audit",
            "",
            "## What V6.1 Improves",
            "",
            "- Rejects fake evidence comparison.",
            "- Rejects human-route labels without responsibility boundary.",
            "- Rejects hidden automatic resolution.",
            "- Rejects omission of minority risk.",
            "- Rejects tampered resolution hashes.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Cases are synthetic.",
            "- There is no real arbitration process.",
            "- Human resolution is still a route label.",
            "",
        ]
    )


def build_claims() -> str:
    return "\n".join(
        [
            "# V6.1 Claims",
            "",
            "## Supported",
            "",
            "- V6.1 rejects tested fake-comparison, route-label, hidden-auto, minority-risk, and hash-tampering false positives.",
            "",
            "## Not Supported",
            "",
            "- Real arbitration.",
            "- Legal adjudication.",
            "- Deployment governance.",
            "",
        ]
    )


def build_limitations() -> str:
    return "\n".join(
        [
            "# V6.1 Limitations",
            "",
            "- Attacks are finite and hand-designed.",
            "- Resolution hashes are consistency checks, not security.",
            "- No real expert process is modeled.",
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
    parser.add_argument("--config", default="configs/v61_hardening.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['resolver']}: hardening_v61_gated_score={row['hardening_v61_gated_score']:.3f}")


if __name__ == "__main__":
    main()

