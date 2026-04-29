from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .case_schema import load_cases
from .metrics_v6 import aggregate_records, score_resolution
from .report_v6 import build_claims, build_limitations, build_report, build_self_audit
from .resolvers import make_resolver


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


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cases = load_cases(config["cases_path"])
    gates = config["gates"]
    records: list[dict[str, Any]] = []
    resolution_log: list[dict[str, Any]] = []

    for resolver_name in config["resolvers"]:
        resolver = make_resolver(resolver_name)
        for case in cases:
            record = resolver.resolve(case)
            resolution_log.append(record)
            scores = score_resolution(case, record)
            records.append({"resolver": resolver_name, "case_id": case["case_id"], **scores})

    summary = aggregate_records(records, gates)
    outputs = config["outputs"]
    write_csv(outputs["records"], records)
    write_csv(outputs["summary"], summary)
    Path(outputs["resolution_log"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["resolution_log"]).write_text(json.dumps(resolution_log, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(outputs["report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["report"]).write_text(build_report(summary, gates), encoding="utf-8")
    Path(outputs["self_audit"]).write_text(build_self_audit(summary), encoding="utf-8")
    Path(outputs["claims"]).write_text(build_claims(), encoding="utf-8")
    Path(outputs["limitations"]).write_text(build_limitations(), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_audit.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['resolver']}: gated_v6_score={row['gated_v6_score']:.3f}")


if __name__ == "__main__":
    main()

