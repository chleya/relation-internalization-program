from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .report import markdown_table


COMPONENTS = [
    "answer_match",
    "inspect_match",
    "uncertain_match",
    "audit_links_match",
    "query_sets_match",
    "answers_by_query_match",
]


def _norm(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_raw(path: str | Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    raw: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_path = Path(path)
    if not raw_path.exists():
        return raw
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("solver", "")), str(row.get("seed", "")), str(row.get("case_id", "")))
            raw[key] = row
    return raw


def case_variant(case_id: str, gate: str) -> str:
    match = re.match(r"seed\d+_budgeted_inspect_(.+)", case_id)
    if match:
        return match.group(1)
    match = re.match(r"seed\d+_audit_(.+)", case_id)
    if match:
        return match.group(1)
    match = re.match(r"seed\d+_(.+?)(?:_strict)?$", case_id)
    if match:
        return match.group(1)
    return gate


def inspect_error(expected: dict[str, Any], response: dict[str, Any], inspect_match: float) -> str:
    expected_inspect = _norm(expected.get("inspect", ""))
    actual_inspect = _norm(response.get("inspect", ""))
    if inspect_match >= 1.0:
        return "correct"
    if response.get("parse_error"):
        return "parse_error"
    if not expected_inspect:
        return "no_expected_inspect"
    if expected_inspect == "none" and actual_inspect and actual_inspect != "none":
        return "over_inspect"
    if expected_inspect != "none" and (not actual_inspect or actual_inspect == "none"):
        return "no_inspect"
    if not actual_inspect:
        return "missing_inspect_field"
    return "wrong_variable"


def merged_rows(records: list[dict[str, Any]], raw: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        key = (str(record.get("solver", "")), str(record.get("seed", "")), str(record.get("case_id", "")))
        raw_row = raw.get(key, {})
        expected = raw_row.get("expected", {})
        response = raw_row.get("response", {})
        rows.append(
            {
                **record,
                "variant": case_variant(str(record.get("case_id", "")), str(record.get("gate", ""))),
                "expected": expected,
                "response": response,
                "expected_inspect": expected.get("inspect", ""),
                "actual_inspect": response.get("inspect", ""),
                "expected_uncertain": expected.get("uncertain", ""),
                "actual_uncertain": response.get("uncertain", ""),
                "expected_audit_links": ";".join(_as_list(expected.get("audit_links"))),
                "actual_audit_links": ";".join(_as_list(response.get("audit_links"))),
                "inspect_error": inspect_error(expected, response, _float(record.get("inspect_match"))),
            }
        )
    return rows


def group_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    components = [component for component in COMPONENTS if any(component in row for row in rows)]
    summary: list[dict[str, Any]] = []
    for name in sorted(grouped):
        items = grouped[name]
        out = {
            group_key: name,
            "n": len(items),
            "case_score": mean(_float(item.get("case_score")) for item in items),
        }
        for component in components:
            out[component] = mean(_float(item.get(component)) for item in items)
        summary.append(out)
    return summary


def inspect_error_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row["inspect_error"]) for row in rows)
    total = sum(counts.values()) or 1
    return [
        {"inspect_error": name, "n": count, "rate": count / total}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "solver",
        "seed",
        "case_id",
        "gate",
        "variant",
        "case_score",
        "expected_inspect",
        "actual_inspect",
        "inspect_error",
        "expected_uncertain",
        "actual_uncertain",
        "expected_audit_links",
        "actual_audit_links",
        "answer_match",
        "inspect_match",
        "uncertain_match",
        "audit_links_match",
            "query_sets_match",
            "answers_by_query_match",
            "error",
        ]
    return [
        {field: row.get(field, "") for field in fields}
        for row in rows
        if _float(row.get("case_score")) < 1.0
    ]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_analysis_report(rows: list[dict[str, Any]], title: str) -> str:
    gates = sorted({str(row.get("gate", "")) for row in rows})
    failed_cases_heading = "## 4. Failed Cases"
    sections = [
        f"# {title}",
        "",
        "## 1. Gate Summary",
        "",
        markdown_table(group_summary(rows, "gate")),
    ]
    if len(gates) == 1 and any(str(row.get("variant", "")) != gates[0] for row in rows):
        sections.extend(
            [
                "",
                "## 2. Variant Summary",
                "",
                markdown_table(group_summary(rows, "variant")),
                "",
                "## 3. Inspect Error Summary",
                "",
                markdown_table(inspect_error_summary(rows)),
            ]
        )
    else:
        failed_cases_heading = "## 2. Failed Cases"
    sections.extend(
        [
            "",
            failed_cases_heading,
            "",
            markdown_table(failure_rows(rows)),
            "",
        ]
    )
    return "\n".join(sections)


def analyze(records_path: str | Path, raw_path: str | Path, title: str) -> tuple[str, list[dict[str, Any]]]:
    records = load_records(records_path)
    raw = load_raw(raw_path)
    rows = merged_rows(records, raw)
    return build_analysis_report(rows, title), failure_rows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--title", default="LLM Relation Result Analysis")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    report, failures = analyze(args.records, args.raw, args.title)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.out_csv:
        write_csv(args.out_csv, failures)


if __name__ == "__main__":
    main()
