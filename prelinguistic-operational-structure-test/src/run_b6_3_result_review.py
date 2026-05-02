from __future__ import annotations

import argparse
from pathlib import Path

from .b6_3_structural_necessity.result_review import run_result_review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/b6_3_structural_necessity_summary.csv")
    args = parser.parse_args()
    review = run_result_review(Path(args.summary))
    print(f"b6_3_submit_ready_as_diagnostic={str(review['submit_ready_as_diagnostic']).lower()}")
    print(f"b6_3_trace_necessity_drop={review['mechanism_necessity']['trace']:.3f}")
    print(f"b6_3_candidate_search_drop={review['mechanism_necessity']['candidate_search']:.3f}")


if __name__ == "__main__":
    main()
