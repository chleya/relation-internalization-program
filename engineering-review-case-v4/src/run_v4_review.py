from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .case_schema import load_cases
from .metrics_v4 import aggregate_records, score_review
from .report_v4 import build_claims, build_limitations, build_report, build_self_audit
from .reviewers import make_reviewer


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
    for agent_name in config["agents"]:
        reviewer = make_reviewer(agent_name)
        for case in cases:
            review = reviewer.review(case)
            scores = score_review(case, review)
            records.append(
                {
                    "agent": agent_name,
                    "case_id": case["case_id"],
                    **scores,
                    "recommended_review_status": review["recommended_review_status"],
                }
            )

    summary = aggregate_records(records, gates)
    outputs = config["outputs"]
    write_csv(outputs["records"], records)
    write_csv(outputs["summary"], summary)

    Path(outputs["report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["report"]).write_text(build_report(summary, gates), encoding="utf-8")
    Path(outputs["self_audit"]).write_text(build_self_audit(summary), encoding="utf-8")
    Path(outputs["claims"]).write_text(build_claims(), encoding="utf-8")
    Path(outputs["limitations"]).write_text(build_limitations(), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v4_review.yaml")
    args = parser.parse_args()

    summary = run(args.config)
    for row in summary:
        print(f"{row['agent']}: gated_v4_score={row['gated_v4_score']:.3f}")


if __name__ == "__main__":
    main()

