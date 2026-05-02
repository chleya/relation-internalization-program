from __future__ import annotations

import argparse
from pathlib import Path

from .b6_2_hardening.result_review import run_result_review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b6_2_hardening_summary.csv")
    parser.add_argument("--records", default="results/b6_2_hardening_records.csv")
    parser.add_argument("--metrics", default="results/b6_2_hardening_metrics.json")
    parser.add_argument("--json-output", default="results/b6_2_result_review.json")
    parser.add_argument("--report-output", default="reports/B6_2_RESULT_REVIEW.md")
    args = parser.parse_args()

    review = run_result_review(
        summary_path=Path(args.summary),
        records_path=Path(args.records),
        metrics_path=Path(args.metrics),
        json_output_path=Path(args.json_output),
        report_output_path=Path(args.report_output),
    )
    wrong_trace = review["wrong_trace_audit"]
    print(f"b6_2_review_wrong_trace_score={wrong_trace['b62_policy_score']:.3f}")
    print(f"b6_2_review_wrong_trace_gain_over_mask_only={wrong_trace['gain_over_mask_only']:.3f}")
    print(f"b6_2_review_missing_focus_conditions={len(review['missing_focus_conditions'])}")


if __name__ == "__main__":
    main()
