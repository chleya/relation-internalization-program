from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .case_schema import load_cases
from .metrics_v5 import aggregate_records, score_event
from .report_v5 import build_claims, build_limitations, build_report, build_self_audit
from .shells import make_shell


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
    decision_log: list[dict[str, Any]] = []
    for shell_name in config["shells"]:
        shell = make_shell(shell_name)
        for case in cases:
            event = shell.process(case)
            decision_log.append(event)
            scores = score_event(case, event)
            records.append(
                {
                    "shell": shell_name,
                    "case_id": case["case_id"],
                    "route": event.get("route", ""),
                    "auto_approved": event.get("auto_approved", ""),
                    **scores,
                }
            )

    summary = aggregate_records(records, gates)
    outputs = config["outputs"]
    write_csv(outputs["records"], records)
    write_csv(outputs["summary"], summary)
    Path(outputs["decision_log"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["decision_log"]).write_text(json.dumps(decision_log, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(outputs["report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["report"]).write_text(build_report(summary, gates), encoding="utf-8")
    Path(outputs["self_audit"]).write_text(build_self_audit(summary), encoding="utf-8")
    Path(outputs["claims"]).write_text(build_claims(), encoding="utf-8")
    Path(outputs["limitations"]).write_text(build_limitations(), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v5_governance.yaml")
    args = parser.parse_args()
    rows = run(args.config)
    for row in rows:
        print(f"{row['shell']}: gated_v5_score={row['gated_v5_score']:.3f}")


if __name__ == "__main__":
    main()

