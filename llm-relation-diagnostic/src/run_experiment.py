from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .cases import make_case_set
from .metrics import evaluate_solver, summarize_records
from .report import build_report, build_self_audit
from .solvers import make_solver


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


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


def label_output_path(path: str | Path, label: str) -> str:
    if not label:
        return str(path)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    output = Path(path)
    return str(output.with_name(f"{output.stem}_{safe_label}{output.suffix}"))


def labeled_outputs(outputs: dict[str, str], label: str) -> dict[str, str]:
    return {name: label_output_path(path, label) for name, path in outputs.items()}


def run(
    config_path: str | Path,
    solver_override: list[str] | None = None,
    base_url: str = "http://127.0.0.1:8083",
    model: str = "local",
    label: str = "",
) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    solver_names = solver_override or list(config["solvers"])
    records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    case_profile = config.get("case_profile", "base")
    case_set = config.get("case_set", "core")

    for solver_name in solver_names:
        solver = make_solver(solver_name, base_url=base_url, model=model)
        for seed in config["seeds"]:
            cases = make_case_set(int(seed), profile=case_profile, case_set=case_set)
            solver_records, solver_raw = evaluate_solver(solver, cases, int(seed))
            records.extend(solver_records)
            raw_records.extend(solver_raw)

    summary = summarize_records(records, config["gates"])
    outputs = labeled_outputs(config["outputs"], label)
    write_csv(outputs["records"], records)
    write_csv(outputs["summary"], summary)
    write_jsonl(outputs["raw"], raw_records)
    Path(outputs["report"]).parent.mkdir(parents=True, exist_ok=True)
    Path(outputs["report"]).write_text(build_report(summary), encoding="utf-8")
    Path(outputs["self_audit"]).write_text(build_self_audit(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--solvers", default="", help="comma-separated solver override, e.g. llama_cpp")
    parser.add_argument("--base-url", default="http://127.0.0.1:8083")
    parser.add_argument("--model", default="local")
    parser.add_argument("--label", default="", help="suffix output files, e.g. qwen15b_smoke")
    args = parser.parse_args()

    solver_override = [item.strip() for item in args.solvers.split(",") if item.strip()] or None
    rows = run(args.config, solver_override=solver_override, base_url=args.base_url, model=args.model, label=args.label)
    for row in rows:
        print(f"{row['solver']}: llm_relation_gated_score={row['llm_relation_gated_score']:.3f}")


if __name__ == "__main__":
    main()
