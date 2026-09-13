from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .g1_2_feature_induction import run_feature_induction
from .g1_metrics import RECORD_FIELDS, SUMMARY_FIELDS


def run_g1_2(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    return run_feature_induction(config, seed)


def write_g1_2_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    out_dir = Path("results/g1_2_feature_induction")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(out_dir / "records.csv", records, RECORD_FIELDS)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/G1_2_FEATURE_INDUCTION_RESULTS.md").write_text(build_report(metrics), encoding="utf-8")


def build_report(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# G1.2 Feature Induction Results",
            "",
            "## Summary",
            "G1.2 reduces hand-scaffolded formulas by searching sparse feature programs over primitive interaction features.",
            "",
            "## Metrics",
            f"- g1_2_mean_score = {float(metrics['g1_2_mean_score']):.3f}",
            f"- g1_2_ood_score = {float(metrics['g1_2_ood_score']):.3f}",
            f"- feature_program_complexity = {int(metrics['feature_program_complexity'])}",
            f"- feature_search_size = {int(metrics['feature_search_size'])}",
            f"- feedback_feature_drop = {float(metrics['feedback_feature_drop']):.3f}",
            f"- compression_feature_drop = {float(metrics['compression_feature_drop']):.3f}",
            f"- gain_over_random_feature_program = {float(metrics['gain_over_random_feature_program']):.3f}",
            f"- oracle_gap = {float(metrics['oracle_gap']):.3f}",
            f"- mask_f1 = {float(metrics['mask_f1']):.3f}",
            "",
            "## Claim Boundary",
            "G1.2 remains a toy feature-induction diagnostic. It reduces formula hand-design but still uses a hand-defined primitive feature vocabulary.",
            "",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
