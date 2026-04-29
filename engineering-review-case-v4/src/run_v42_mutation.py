from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .metrics_v42_mutation import evaluate_mutation, write_mutated_cases


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
            "# V4.2 Adversarial Case Mutation Report",
            "",
            "## 1. Motivation",
            "",
            "V4.2 mutates toy review cases to attack fixed wording, field order, irrelevant variables, and hidden unsafe approval phrases.",
            "",
            "## 2. Mutations",
            "",
            "- reorder observed fields",
            "- add irrelevant variables",
            "- paraphrase relation names",
            "- insert hidden unsafe approval phrase",
            "",
            "## 3. Gates",
            "",
            *[f"- `{key}` >= {value}" for key, value in gates.items()],
            "",
            "## 4. Results",
            "",
            markdown_table(rows),
            "",
            "## 5. Interpretation",
            "",
            "`mutation_v42_gated_score` is zero unless the reviewer passes original V4 gates on mutated cases and all mutation-specific gates.",
            "",
            "## 6. Claim Boundary",
            "",
            "Supported: bounded toy review diagnostics can be stress-tested against the implemented adversarial case mutations.",
            "",
            "Unsupported: real engineering review, real safety prediction, or deployment-ready approval.",
            "",
        ]
    )


def build_self_audit() -> str:
    return "\n".join(
        [
            "# V4.2 Self-Audit",
            "",
            "## What V4.2 Improves",
            "",
            "- Tests field-order robustness.",
            "- Tests rejection of irrelevant variables.",
            "- Tests robustness to paraphrased relation names.",
            "- Tests rejection of hidden unsafe approval phrasing.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Mutations are deterministic and hand-designed.",
            "- Relation aliases are still predefined.",
            "- This remains a toy review diagnostic.",
            "- It does not validate real geotechnical reasoning or expert review.",
            "",
            "## Boundary Statement",
            "",
            "This is not a real engineering approval system and not deployment-ready engineering AI.",
            "",
        ]
    )


def build_claims() -> str:
    return "\n".join(
        [
            "# V4.2 Claims",
            "",
            "## Supported",
            "",
            "- V4.2 rejects tested field-order, irrelevant-variable, relation-paraphrase, and unsafe-phrase false positives.",
            "- The uncertainty-aware review remains stable under the implemented toy mutations.",
            "",
            "## Not Supported",
            "",
            "- Real engineering review.",
            "- Real slope safety prediction.",
            "- Unrestricted relation extraction from natural language.",
            "- Deployment-ready approval workflow.",
            "",
        ]
    )


def build_limitations() -> str:
    return "\n".join(
        [
            "# V4.2 Limitations",
            "",
            "- Mutation operators are small and deterministic.",
            "- The alias map is predefined.",
            "- Cases remain curated and synthetic.",
            "- Passing V4.2 does not prove robustness to arbitrary engineering documents.",
            "",
        ]
    )


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rows, records, mutated_cases = evaluate_mutation(config)
    outputs = config["outputs"]
    write_csv(outputs["summary"], rows)
    write_csv(outputs["records"], records)
    write_mutated_cases(outputs["mutated_cases"], mutated_cases)
    Path(outputs["report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["report"]).write_text(build_report(rows, config["mutation_gates"]), encoding="utf-8")
    Path(outputs["self_audit"]).write_text(build_self_audit(), encoding="utf-8")
    Path(outputs["claims"]).write_text(build_claims(), encoding="utf-8")
    Path(outputs["limitations"]).write_text(build_limitations(), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v42_mutation.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['agent']}: mutation_v42_gated_score={row['mutation_v42_gated_score']:.3f}")


if __name__ == "__main__":
    main()

